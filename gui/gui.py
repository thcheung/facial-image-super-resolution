import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
import tkinter.filedialog as filedialog
from resolve import resolver
import os
from subprocess import Popen, PIPE
import shutil

global p
p = Popen(['all.exe'], shell=True, stdout=PIPE, stdin=PIPE)
result = p.stdout.readline().strip()
print(result)
result = p.stdout.readline().strip()
print(result)


def clear():
    print('clear')
    try:
        lr.destroy()
    except NameError:
        pass
    try:
        hr.destroy()
    except NameError:
        pass
    try:
        lr_size.destroy()
    except NameError:
        pass
    try:
        hr_size.destroy()
    except NameError:
        pass


def allMethods():
    print('all')
    subwindow = tk.Toplevel()
    subwindow.title("All methods")
    subwindow.iconbitmap("logo.ico")
    subwindow.geometry("1180x240")

    lrFrame = tk.LabelFrame(subwindow, text=" LR ", width=180, height=180)
    lrFrame.grid(row=0, column=0, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    lrFrame.pack_propagate(0)
    im_lr = Image.open(filename)
    width, height = im_lr.size
    global lr_im_size
    try:
        lr_im_size.destroy()
    except NameError:
        pass
    lr_im_size = tk.Label(lrFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    lr_im_size.pack(side = 'bottom')
    lr_im_size.config(font=(None, 10))
    scale = 128/height
    im = ImageTk.PhotoImage(im_lr.resize((int(width*scale),int(height*scale))))
    global lr_im
    try:
        lr_im.destroy()
    except NameError:
        pass
    lr_im = tk.Label(lrFrame, image = im)
    lr_im.image = im
    lr_im.pack(fill="none", expand=True)
    print (filename)

    
    hrPCAFrame = tk.LabelFrame(subwindow, text=" PCA ", width=180, height=180)
    hrPCAFrame.grid(row=0, column=1, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    hrPCAFrame.pack_propagate(0)

    hrLLEFrame = tk.LabelFrame(subwindow, text=" LLE ", width=180, height=180)
    hrLLEFrame.grid(row=0, column=2, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    hrLLEFrame.pack_propagate(0)

    hrSSRFrame = tk.LabelFrame(subwindow, text=" SSR ", width=180, height=180)
    hrSSRFrame.grid(row=0, column=3, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    hrSSRFrame.pack_propagate(0)

    hrCNNFrame = tk.LabelFrame(subwindow, text=" CNN ", width=180, height=180)
    hrCNNFrame.grid(row=0, column=4, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    hrCNNFrame.pack_propagate(0)

    hrGANFrame = tk.LabelFrame(subwindow, text=" GAN ", width=180, height=180)
    hrGANFrame.grid(row=0, column=5, columnspan=1, sticky="E", padx=5, pady=5, ipadx=0, ipady=0)
    hrGANFrame.pack_propagate(0)
    
    global hr_cnn
    global hr_gan
    global hr_pca
    global hr_lle
    global hr_ssr
    try:
        hr_cnn.destroy()
    except NameError:
        pass
    try:
        hr_gan.destroy()
    except NameError:
        pass
    try:
        hr_pca.destroy()
    except NameError:
        pass
    try:
        hr_lle.destroy()
    except NameError:
        pass
    try:
        hr_ssr.destroy()
    except NameError:
        pass

    path_lr = filename
    path_sr = 'results/' + os.path.split(filename)[1]

    writeParameters2('PCA')
    value = str(1) +'\n'
    value = bytes(value, 'UTF-8')
    p.stdin.write(value)
    p.stdin.flush()
    result = p.stdout.readline().strip()
    print(result)
    hr_size = tk.Label(hrPCAFrame,text = 'Size: ' + str(width*scaleFactor.get()) + 'x' + str(height*scaleFactor.get()))
    hr_size.pack(side = 'bottom')
    hr_size.config(font=(None, 10))
    
    path_pca = 'results/pca_' + os.path.split(filename)[1]
    shutil.copy(path_sr,path_pca)
    im_pca=Image.open(path_pca)
    width, height = im_pca.size
    im = ImageTk.PhotoImage(im_pca.resize((int(width),int(height))))
    hr_pca = tk.Label(hrPCAFrame, image = im)
    hr_pca.image = im
    hr_pca.pack(fill="none", expand=True) 

    writeParameters2('LLE')
    value = str(1) +'\n'
    value = bytes(value, 'UTF-8')
    p.stdin.write(value)
    p.stdin.flush()
    result = p.stdout.readline().strip()
    print(result)
    
    path_lle = 'results/lle_' + os.path.split(filename)[1]
    shutil.copy(path_sr,path_lle)
    im_pca=Image.open(path_lle)
    
    hr_size = tk.Label(hrLLEFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    hr_size.pack(side = 'bottom')
    hr_size.config(font=(None, 10))
    
    im_lle=Image.open(path_sr)
    width, height = im_lle.size
    im = ImageTk.PhotoImage(im_lle.resize((int(width),int(height))))
    hr_lle = tk.Label(hrLLEFrame, image = im)
    hr_lle.image = im
    hr_lle.pack(fill="none", expand=True) 

    writeParameters2('SSR')
    value = str(1) +'\n'
    value = bytes(value, 'UTF-8')
    p.stdin.write(value)
    p.stdin.flush()
    result = p.stdout.readline().strip()       
    print(result)

    path_ssr = 'results/ssr_' + os.path.split(filename)[1]
    shutil.copy(path_sr,path_ssr)
    im_pca=Image.open(path_ssr)
    
    hr_size = tk.Label(hrSSRFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    hr_size.pack(side = 'bottom')
    hr_size.config(font=(None, 10))
    im_ssr=Image.open(path_sr)
    width, height = im_ssr.size
    im = ImageTk.PhotoImage(im_ssr.resize((int(width),int(height))))
    hr_ssr = tk.Label(hrSSRFrame, image = im)
    hr_ssr.image = im
    hr_ssr.pack(fill="none", expand=True)

    resolver(path_lr,path_sr,scaleFactor.get(),dataset.get(),'CNN')
    hr_size = tk.Label(hrCNNFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    hr_size.pack(side = 'bottom')
    hr_size.config(font=(None, 10))
    im_cnn=Image.open(path_sr)
    im = ImageTk.PhotoImage(im_cnn.resize((int(width),int(height))))
    hr_cnn = tk.Label(hrCNNFrame, image = im)
    hr_cnn.image = im
    hr_cnn.pack(fill="none", expand=True)     

    resolver(path_lr,path_sr,scaleFactor.get(),dataset.get(),'GAN')
    hr_size = tk.Label(hrGANFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    hr_size.pack(side = 'bottom')
    hr_size.config(font=(None, 10))
    im_gan=Image.open(path_sr)
    im = ImageTk.PhotoImage(im_gan.resize((int(width),int(height))))
    hr_gan = tk.Label(hrGANFrame, image = im)
    hr_gan.image = im
    hr_gan.pack(fill="none", expand=True)


def load():
    print('load')
    global filename
    filename =  filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    im_lr = Image.open(filename)
    width, height = im_lr.size
    global lr_size
    try:
        lr_size.destroy()
    except NameError:
        pass
    lr_size = tk.Label(lrFrame,text = 'Size: ' + str(width) + 'x' + str(height))
    lr_size.pack(side = 'bottom')
    lr_size.config(font=(None, 10))
    scale = 128/height
    im = ImageTk.PhotoImage(im_lr.resize((int(width*scale*1.5),int(height*scale*1.5))))
    global lr
    try:
        lr.destroy()
    except NameError:
        pass
    lr = tk.Label(lrFrame, image = im)
    lr.image = im
    lr.pack(fill="none", expand=True)
    print (filename)


def writeParameters():
    fo = open("parameters.txt", "w")
    fo.write('#Parameters' + '\n')
    fo.write('upscale ' + str(scaleFactor.get()) + '\n')
    fo.write('dataset ' + str(dataset.get()) + '\n')
    fo.write('method ' + method.get() + '\n')
    fo.write('test_lr ' + filename +'\n')
    fo.write('test_sr' + ' results/' + os.path.split(filename)[1])
    fo.close()

def writeParameters2(myMethod):
    fo = open("parameters.txt", "w")
    fo.write('#Parameters' + '\n')
    fo.write('upscale ' + str(scaleFactor.get()) + '\n')
    fo.write('dataset ' + str(dataset.get()) + '\n')
    fo.write('method ' + myMethod + '\n')
    fo.write('test_lr ' + filename +'\n')
    fo.write('test_sr' + ' results/' + os.path.split(filename)[1])
    fo.close()

def start():
    print('start')
    global hr
    try:
        hr.destroy()
    except NameError:
        pass
    
    if (method.get()=='CNN'):
        path_lr = filename;
        path_sr = 'results/' + os.path.split(filename)[1]
        resolver(path_lr,path_sr,scaleFactor.get(),dataset.get(),'CNN')

        
    if (method.get()=='GAN'):
        path_lr = filename;
        path_sr = 'results/' + os.path.split(filename)[1]
        resolver(path_lr,path_sr,scaleFactor.get(),dataset.get(),'GAN')


    if (method.get()=='PCA'):
        writeParameters()
        value = str(1) +'\n'
        value = bytes(value, 'UTF-8')
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        print(result)
 

    if (method.get()=='LLE'):
        writeParameters()
        value = str(1) +'\n'
        value = bytes(value, 'UTF-8')
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()
        print(result)
 
    if (method.get()=='SSR'):
        writeParameters()
        value = str(1) +'\n'
        value = bytes(value, 'UTF-8')
        p.stdin.write(value)
        p.stdin.flush()
        result = p.stdout.readline().strip()       
        print(result)
 
    if (method.get()!='ALL'):
        path_sr = 'results/' + os.path.split(filename)[1]
        im_hr=Image.open(path_sr)
        width, height = im_hr.size
        global hr_size
        try:
            hr_size.destroy()
        except NameError:
            pass
        hr_size = tk.Label(hrFrame,text = 'Size: ' + str(width) + 'x' + str(height))
        hr_size.pack(side = 'bottom')
        hr_size.config(font=(None, 10))
        im = ImageTk.PhotoImage(im_hr.resize((int(width*1.5),int(height*1.5))))
        hr = tk.Label(hrFrame, image = im)
        hr.image = im
        hr.pack(fill="none", expand=True)
    else:
        allMethods()

def save():
    print('save')
    filename_hr =  filedialog.asksaveasfilename(initialdir = ".",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
    path_sr = 'results/' + os.path.split(filename)[1]
    shutil.copy(path_sr, filename_hr)


window = tk.Tk()
window.title("SuperResolution")
window.iconbitmap("logo.ico")
window.geometry("700x700")



title = tk.Label(window, text = 'Machine Learning for Facial Image Super-resolution')
title.grid(column=0, row=0, sticky = 'NEWS',pady=5)
title.config(font=(None, 14))

name = tk.Label(window, text = 'CHEUNG Tsun Hin, 15083269d@connect.polyu.hk      ')
name.grid(column=0, row=1, sticky = 'NES')
name.config(font=(None, 8))

upperFrame = tk.Frame(window, width=680, height=200)
upperFrame.grid(column=0, row=2,sticky = 'NEWS', pady=20)

lrFrame = tk.LabelFrame(upperFrame, text="LR Image ", width=300, height=300)
lrFrame.grid(row=0, column=0, columnspan=1, sticky="E", padx=20, pady=5, ipadx=0, ipady=0)
lrFrame.pack_propagate(0)


hrFrame = tk.LabelFrame(upperFrame, text=" HR Image ", width=300, height=300)
hrFrame.grid(row=0, column=1, columnspan=1, sticky="E", padx=20, pady=5, ipadx=0, ipady=0)
hrFrame.pack_propagate(0)


middleFrame = tk.Frame(window, width=680, height=150)
middleFrame.grid(column=0, row=3,sticky = 'NEWS', pady=20)

s = ttk.Style()
s.configure('my.TButton', font=('None', 12))

b0 = ttk.Button(middleFrame, text="Load", command=load, width=10)
b0.config(style='my.TButton')
b0.grid(row=0, column=0, sticky="E", padx=25, pady=5, ipadx=10, ipady=0)

b1 = ttk.Button(middleFrame, text="Clear", command=clear, width=10)
b1.config(style='my.TButton')
b1.grid(row=0, column=1, sticky="E", padx=25, pady=5, ipadx=10, ipady=0)

b2 = ttk.Button(middleFrame, text="Start", command=start, width=10)
b2.config(style='my.TButton')
b2.grid(row=0, column=2, sticky="E", padx=25, pady=5, ipadx=10, ipady=0)

b3 = ttk.Button(middleFrame, text="Save", command=save, width=10)
b3.config(style='my.TButton')
b3.grid(row=0, column=3, sticky="E", padx=25, pady=5, ipadx=10, ipady=0)

buttomFrame = tk.Frame(window, width=680)
buttomFrame.grid(column=0, row=4,sticky = 'NEWS', pady=20)

leftFrame = tk.Frame(buttomFrame, width=300, height=250)
leftFrame.grid(column=0, row=0,sticky = 'NEWS')

scaleFrame = tk.LabelFrame(leftFrame, text="Upscaling Factor",font= (None,10), width=340)
scaleFrame.grid(column=0, row=0,  sticky = 'NEWS',padx=20, pady=5, ipadx=35, ipady=0)
scaleFactor = tk.IntVar()
R1 = tk.Radiobutton(scaleFrame, text = "4x",font= (None,10), variable = scaleFactor, value = 4).grid(column=0, row=0, sticky = 'W')
R2 = tk.Radiobutton(scaleFrame, text = "8x",font= (None,10), variable = scaleFactor, value = 8).grid(column=0, row=1, sticky = 'W')
scaleFactor.set(8)

datasetFrame = tk.LabelFrame(leftFrame, text="Training Dataset",font= (None,10), width=340)
datasetFrame.grid(column=0, row=1,  sticky = 'NEWS',padx=20, pady=5, ipadx=35, ipady=0)
dataset = tk.IntVar()
D1 = tk.Radiobutton(datasetFrame, text = "Constrained Dataset",font= (None,10), variable = dataset, value = 1).grid(column=0, row=0, sticky = 'W')
D2 = tk.Radiobutton(datasetFrame, text = "Unconstrained Dataset",font= (None,10), variable = dataset, value = 2).grid(column=0, row=1, sticky = 'W')
dataset.set(1)

methodFrame = tk.LabelFrame(buttomFrame, text="Algorithm",font= (None,10), width=300, height=250)
methodFrame.grid(column=1, row=0, sticky = 'NEWS', padx=20, pady=5, ipadx=50, ipady=0)

method = tk.StringVar()
A1 = tk.Radiobutton(methodFrame, text = "Eigentransformation (PCA)",font= (None,10), variable = method, value = 'PCA')
A1.grid(column=0, row=0, sticky = 'W')
A2 = tk.Radiobutton(methodFrame, text = "Neighbour Embedding (LLE)",font= (None,10), variable = method, value = 'LLE')
A2.grid(column=0, row=1, sticky = 'W')
A3 = tk.Radiobutton(methodFrame, text = "Smooth Sparse Representation (SSR)",font= (None,10), variable = method, value = 'SSR')
A3.grid(column=0, row=2, sticky = 'W')
A4 = tk.Radiobutton(methodFrame, text = "Convolutional Neural Network (CNN)",font= (None,10), variable = method, value = 'CNN')
A4.grid(column=0, row=3, sticky = 'W')
A5 = tk.Radiobutton(methodFrame, text = "Generative Adversarial Networks (GAN)",font= (None,10), variable = method, value = 'GAN')
A5.grid(column=0, row=4, sticky = 'W')
A6 = tk.Radiobutton(methodFrame, text = "All methods",font= (None,10), variable = method, value = 'ALL')
A6.grid(column=0, row=5, sticky = 'W')
method.set("CNN")
window.mainloop()


