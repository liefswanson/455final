from tkinter import filedialog
from matplotlib.image import imread, imsave
from jinja2 import Template
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from tkinter import *

chosen_file = ""
chosen_effect = ""

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

root.title("Select an effect to apply")

# Add a grid
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)
mainframe.pack(pady = 100, padx = 100)

# Create a Tkinter variable
tkvar = StringVar(root)

# Dictionary with options
choices = {'Brighten x2', 'Edge Detection'}
tkvar.set('Select an effect...') # set the default option

popupMenu = OptionMenu(mainframe, tkvar, *choices)
Label(mainframe, text="Choose a dish").grid(row = 1, column = 1)
popupMenu.grid(row = 2, column =1)

# on change dropdown value
def change_dropdown(*args):
    root.chosen_effect = tkvar.get()
    root.destroy()

# link function to change dropdown
tkvar.trace('w', change_dropdown)

root.mainloop()

chosen_file = root.filename
chosen_effect = root.chosen_effect

if chosen_effect == '' or chosen_file == '':
    exit()

print(chosen_file)
print(chosen_effect)


def test_brighten(img):

    fn = numpy.vectorize(brighten_vectorized)

    cpu_render = fn(img)
    gpu_render = numpy.zeros_like(img, dtype=numpy.uint8)

    module = SourceModule("""
    __global__ void brighten(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x;
        const int col = threadIdx.y;
        const int chan = threadIdx.z;

        const int px = img[row,col,chan];
        const int temp = px*2;

        if (0 > temp){
            dest[row,col,chan] = 0;
            return;
        }
        if (255 < temp){
            dest[row,col,chan] = 255;
            return;
        }
        dest[row,col,chan] = temp;
    }
    """)

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("brighten")
    gpu_brighten(drv.Out(dest), drv.In(img),
                block=img.shape, grid=(1,1))

    return (cpu_render, gpu_render)


def brighten_vectorized(px):
    temp = px * 2
    return clamp(0, 255, temp)


def clamp(low, high, val):
    if low > val:
        return low
    if high < val:
        return high
    return val

def test_edge_detection(img):
    cpu_render = []
    gpu_render = []
    return (cpu_render, gpu_render)


switch = {
    'Edge Detection': test_edge_detection,
    'Brighten x2': test_brighten
}

img = imread(chosen_file)
cpu_render, gpu_render = switch[chosen_effect](img)
imsave('/tmp/cpu_render.png', cpu_render)
imsave('/tmp/gpu_render.png', cpu_render)

from tkinter import *
from PIL import Image, ImageTk

class ImgFrame(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.original = Image.open('/tmp/cpu_render.png')
        self.image = ImageTk.PhotoImage(self.original)
        self.display = Canvas(self, bd=0, highlightthickness=0)
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")
        self.display.grid(row=0, sticky=W+E+N+S)
        self.pack(fill=BOTH, expand=1)
        self.bind("<Configure>", self.resize)

    def resize(self, event):
        w, h = self.original.size
        ew, eh = event.width, event.height

        ratio = w/h

        if ew < eh * ratio:
            size = (round(eh*ratio), eh)
        else:
            size = (ew, round(ew/ratio))

        resized = self.original.resize(size,Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
        self.display.delete("IMG")
        self.display.create_image(0, 0, image=self.image, anchor=NW, tags="IMG")

root = Tk()
app = ImgFrame(root)
app.mainloop()