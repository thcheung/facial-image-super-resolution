from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Generator_4x,Generator_8x, Discriminator
from data import get_training_set, get_test_set
import torchvision.models as models
import torchvision
import random

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)


def train(epoch,batch_size,training_data_loader,device,mse_criterion,bce_criterion,g_optimizer,g_model,d_optimizer,d_model):
    epoch_g_loss = 0
    epoch_d_loss = 0	
    y_real=torch.ones(batch_size,1)
    y_fake=torch.zeros(batch_size,1)
    y_real = y_real.to(device)
    y_fake = y_fake.to(device)	
    for iteration, batch in enumerate(training_data_loader, 1):
        image_lr, image_hr = batch[0].to(device), batch[1].to(device)
        sd = random.uniform(0.0, 0.0001)
        noise = GaussianNoise(sd)        
        image_lr = noise(image_lr)       
        d_model.zero_grad()
        D_real=d_model(noise(image_hr))
        D_real_loss=bce_criterion(D_real,y_real)

        fake_image=g_model(image_lr)
        D_fake=d_model(fake_image)
        D_fake_loss=bce_criterion(D_fake,y_fake)

        D_loss=D_real_loss+D_fake_loss
        D_loss.backward(retain_graph=True)
        d_optimizer.step()
        epoch_d_loss += D_loss.item()
		
        g_model.zero_grad()
        vgg = torchvision.models.vgg19(pretrained=True)		
        feature_extractor = nn.Sequential(*list(vgg.features)[:(11+1)]).to(device).eval()
        G_loss = mse_criterion(g_model(image_lr), image_hr) + 0.006 * mse_criterion(feature_extractor(g_model(image_lr)), feature_extractor(image_hr)) + 0.001*bce_criterion(D_fake,y_real)
        #G_loss = mse_criterion(g_model(image_lr), image_hr)
        epoch_g_loss += G_loss.item()
        G_loss.backward()
        g_optimizer.step()
        if (iteration%1000==0):
            print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), G_loss.item()))
            print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), D_loss.item()))		
        torch.cuda.empty_cache()
    print("===> Epoch {} Complete: Avg. G_Loss: {:.4f}".format(epoch, epoch_g_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. D_Loss: {:.4f}".format(epoch, epoch_d_loss / len(training_data_loader)))

def test(testing_data_loader, device, g_optimizer, mse_criterion,g_model):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            image_lr, image_hr = batch[0].to(device), batch[1].to(device)
            prediction = g_model(image_lr)
            mse = mse_criterion(prediction, image_hr)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch,g_model,d_model,offset):

    if (epoch%10==0):
        model_out_path = "Drive/nn/logs/g_model_epoch_{}.pth".format(epoch+offset)
        torch.save(g_model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        
        model_out_path = "Drive/nn/logs/d_model_epoch_{}.pth".format(epoch+offset)
        torch.save(d_model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

	
def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
	parser.add_argument('--upscale_factor', type=int,default=8, help="super resolution upscale factor")
	parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
	parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
	parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
	parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
	parser.add_argument('--d_lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
	parser.add_argument('--cuda', type=bool,default=True, help='use cuda?')
	parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
	parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
	parser.add_argument('--offset_g',type=int, default=0)
	parser.add_argument('--offset_d',type=int, default=0)	
	parser.add_argument('--g_model',type=str, default="Drive/nn/logs/g_model_epoch_0.pth")
	parser.add_argument('--d_model',type=str, default="Drive/nn/logs/d_model_epoch_0.pth")
	opt = parser.parse_args()

	print(opt)

	if opt.cuda and not torch.cuda.is_available():
		raise Exception("No GPU found, please run without --cuda")

	torch.manual_seed(opt.seed)

	device = torch.device("cuda" if opt.cuda else "cpu")

	print('===> Loading datasets')
	train_set = get_training_set(opt.upscale_factor)
	test_set = get_test_set(opt.upscale_factor)
	training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
	testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

	print('===> Building model')
	if opt.upscale_factor==4:
		g_model = Generator_4x().to(device)
	if opt.upscale_factor==8:
		g_model = Generator_8x().to(device)

	d_model = Discriminator().to(device)
	
	if opt.offset_g!=0:
		g_model.load_state_dict(torch.load(opt.g_model))
	if opt.offset_d!=0:
		d_model.load_state_dict(torch.load(opt.d_model))
		
	mse_criterion = nn.MSELoss()
	bce_criterion = nn.BCELoss()
	
	g_optimizer = optim.Adam(g_model.parameters(), lr=opt.g_lr)
	d_optimizer = optim.Adam(d_model.parameters(), lr=opt.d_lr)
	
	offset = opt.offset_g
	
	for epoch in range(1, opt.nEpochs + 1):
		train(epoch,opt.batchSize,training_data_loader,device,mse_criterion,bce_criterion,g_optimizer,g_model,d_optimizer,d_model)
		test(testing_data_loader,device,g_optimizer,mse_criterion,g_model)
		checkpoint(epoch,g_model,d_model,offset)
		
if __name__ == '__main__':
      # execute only if run as a script
      main()
