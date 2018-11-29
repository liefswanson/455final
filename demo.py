from tkinter import filedialog
from math import ceil
import math
from matplotlib.image import imread, imsave
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from jinja2 import Template
import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import SourceModule
from tkinter import *
from PIL import Image, ImageTk


def test_chiaroscuro(img):
    return(cpu_chiaroscuro, gpu_chiaroscuro(img))


def cpu_chiaroscuro(img):
    result = numpy.zeros_like(img)
    height, width, depth = img.shape
    for row in range(height):
        for col in range(width):
            r = img[row,col, 0] / 255.0
            g = img[row,col, 1] / 255.0
            b = img[row,col, 2] / 255.0

            intensity = 0.21*r + 0.72*g + 0.7*b
            if intensity == 0.0:
                intensity = 1.0

            r = math.pow(r, 1.0/intensity)
            g = math.pow(g, 1.0/intensity)
            b = math.pow(b, 1.0/intensity)

            result[row, col, 0] = round(r * 255.0)
            result[row, col, 1] = round(g * 255.0)
            result[row, col, 2] = round(b * 255.0)

    return result

def gpu_chiaroscuro(img):
    template = Template("""
    __global__ void chiaroscuro(unsigned char *dest, unsigned char *img) {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;

        const int idx = col*{{depth}} + row*{{depth}}*{{height}};

        if (idx+2 > {{width}}*{{height}}*{{depth}}) {
            return;
        }

        const float r = img[idx] / 255.0f;
        const float g = img[idx+1] / 255.0f;
        const float b = img[idx+2] / 255.0f;

        float intensity = 0.21f*r + 0.72*g + 0.07*b;
        if (intensity == 0.0f) {
            intensity = 1.0f;
        }

        const float out_r = __powf(r, 1.0f/intensity);
        const float out_g = __powf(g, 1.0f/intensity);
        const float out_b = __powf(b, 1.0f/intensity);

        dest[idx] = __float2int_rn(out_r*255.0f);
        dest[idx+1] = __float2int_rn(out_g*255.0f);
        dest[idx+2] = __float2int_rn(out_b*255.0f);
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))
    gpu_brighten = module.get_function("chiaroscuro")

    block = (8,8,1)
    grid = (ceil(width/8), ceil(height/8))
    return (gpu_brighten, block, grid)


def gpu_run_effect(gpu_instructions, img):
    effect, block, grid = gpu_instructions
    dest = numpy.zeros_like(img)
    effect(drv.Out(dest), drv.In(img), block=block, grid=grid)
    return dest


def test_gamma_half(img):
    return(cpu_gamma_half, gpu_gamma_half(img))

def gamma_half(px):
    px = px / 255;
    corrected = math.pow(px, 0.5)

    corrected *= 255.0
    return round(corrected)

def cpu_gamma_half(img):
    fn = numpy.vectorize(gamma_half)
    return fn(img)

def gpu_gamma_half(img):
    template = Template("""
    __global__ void gamma_half(unsigned char *dest, unsigned char *img) {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }

        const float px = img[idx] / 255.0f;
        float out = __powf(px, 0.5f);

        dest[idx] = __float2int_rn(out*255.0f);
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("gamma_half")

    block = (8,8,3)
    grid = (ceil(width/8), ceil(height/8))
    return (gpu_brighten, block, grid)



def test_gamma_two(img):
    return(cpu_gamma_two, gpu_gamma_two(img))

def gamma_two(px):
    px = px / 255;
    corrected = math.pow(px, 2.0)
    if px > 1.0:
        px = 1.0
    if px < 0.0:
        px = 0.0
    corrected *= 255.0
    return round(corrected)

def cpu_gamma_two(img):
    fn = numpy.vectorize(gamma_two)
    return fn(img)

def gpu_gamma_two(img):
    template = Template("""
    __global__ void gamma_two(unsigned char *dest, unsigned char *img) {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }

        const float px = img[idx] / 255.0f;
        float out = __powf(px, 2);

        dest[idx] = __float2int_rn(out*255.0f);
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("gamma_two")
    block = (8,8,3)
    grid = (ceil(width/8), ceil(height/8))
    return (gpu_brighten, block, grid)

def test_brighten(img):
    return (cpu_brighten, gpu_brighten(img))

def cpu_brighten(img):
    fn = numpy.vectorize(brighten_vectorized)
    return fn(img)


def gpu_brighten(img):
    template = Template("""
    __global__ void brighten(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};
        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }
        const int px = img[idx];
        const int temp = px*2;

        if (0 > temp){
            dest[idx] = 0;
            return;
        }
        if (255 < temp){
            dest[idx] = 255;
            return;
        }
        dest[idx] = temp;
    }
    """)

    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)

    gpu_brighten = module.get_function("brighten")
    block = (8,8,3)
    grid = (ceil(width/8), ceil(height/8))
    return (gpu_brighten, block, grid)


def brighten_vectorized(px):
    temp = px * 2
    return clamp(0, 255, temp)


def test_edge_detection(img):
    return (edge_detect, gpu_edge_detect(img))

def edge_detect(img):
    result = numpy.zeros_like(img)
    width, height, depth = img.shape
    for row in range(1, width-1):
        for col in range(1, height-1):
            total = 0
            for chan in range(depth):
                total += 8*img[row,col,chan]

                total -= img[row-1,col-1,chan]
                total -= img[row-1,col,chan]
                total -= img[row-1,col+1,chan]

                total -= img[row,col-1,chan]
                total -= img[row,col+1,chan]

                total -= img[row+1,col-1,chan]
                total -= img[row+1,col,chan]
                total -= img[row+1,col+1,chan]

            result[row,col,0] = clamp(0,255,total)
            result[row,col,1] = clamp(0,255,total)
            result[row,col,2] = clamp(0,255,total)

    return result

def gpu_edge_detect(img):
    template = Template("""
    __global__ void edge(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;

        if (row < 1 || row >= {{width}} - 1 ||
            col < 1 || col >= {{height}} - 1) {
            return;
        }

        int total = 0;
        for (int chan = 0; chan < {{depth}}; chan++) {
            const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

            const int down = {{depth}}*{{height}};
            const int right = {{depth}};

            total += 8 * img[idx];

            total -= img[idx - down - right];
            total -= img[idx - down];
            total -= img[idx - down + right];

            total -= img[idx - down];
            total -= img[idx + down];

            total -= img[idx + down - right];
            total -= img[idx + down];
            total -= img[idx + down + right];

        }

        const int chan = threadIdx.z;
        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};

        if (0 > total){
            dest[idx] = 0;
            dest[idx+1] = 0;
            dest[idx+2] = 0;
            return;
        }
        if (255 < total){
            dest[idx] = 255;
            dest[idx+1] = 255;
            dest[idx+2] = 255;
            return;
        }
        dest[idx] = total;
        dest[idx+1] = total;
        dest[idx+2] = total;
    }
    """)


    width, height, depth = img.shape
    module = SourceModule(template.render(width=width, height=height, depth=depth))

    dest = numpy.zeros_like(img)
    gpu_brighten = module.get_function("edge")
    block = (8,8,1)
    grid = (ceil(width/8), ceil(height/8))
    return (gpu_brighten, block, grid)


def clamp(low, high, val):
    if low > val:
        return low
    if high < val:
        return high
    return val

chosen_file = ""
chosen_effect = ""

switch = {
    'Edge Detection': test_edge_detection,
    'Brighten x2': test_brighten,
    'Gamma 2.0': test_gamma_two,
    'Gamma 0.5': test_gamma_half,
    'Chiaroscuro': test_chiaroscuro,
}

###############################################################################################
## UI
###############################################################################################
root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))

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
choices = list(switch.keys())
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


img = imread(chosen_file)

cpu_renderer, gpu_renderer = switch[chosen_effect](img)
imsave('/tmp/cpu_render.png', cpu_renderer(img))
imsave('/tmp/gpu_render.png', gpu_run_effect(gpu_renderer, img))


class ImgFrame(Frame):
    def __init__(self, master, path):
        Frame.__init__(self, master)
        self.columnconfigure(0,weight=1)
        self.rowconfigure(0,weight=1)
        self.original = Image.open(path)
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


width, height, _ = img.shape

ratio = round(width/height)

root = Tk()
root.title(chosen_file + ' ' + chosen_effect)
left = Label(root)
left.pack(side=LEFT)

mid = Label(root)
mid.pack(side=LEFT)

right = Label(root)
right.pack(side=LEFT)

left = ImgFrame(left, '/tmp/cpu_render.png')
mid = ImgFrame(mid, chosen_file)
right = ImgFrame(right, '/tmp/gpu_render.png')

def perf_test(frames, name, cpu_renderer, gpu_renderer, img, root, output):
    cpu_control, gpu_control = build_control(img)

    cpu_total, cpu_control_total = test(frames, name, cpu_renderer, cpu_control, img)
    gpu_total, gpu_control_total = gpu_test(frames, name, gpu_renderer, gpu_control, img)

    f = open('/tmp/gpu_perf_results', 'w')

    f.write(str(cpu_total) + '\n')
    f.write(str(cpu_control_total) + '\n')
    f.write(str(gpu_total) + '\n')
    f.write(str(gpu_control_total) + '\n')

    output.append(cpu_total)
    output.append(cpu_control_total)
    output.append(gpu_total)
    output.append(gpu_control_total)

    root.destroy()

def test(frames, name, fn, ctrl, img):
    fn(img)
    start = timer()
    for _ in range(frames):
        result = fn(img)
    total = timer() - start

    ctrl(img)
    start = timer()
    for _ in range(frames):
        result = ctrl(img)
    control_total = timer() - start

    print(name + ' cpu-test')
    print('%f fps' % (frames / total)) # actual time
    print('%f fps' % (frames / control_total)) # travel time
    print('%f fps' % (frames / (total-control_total))) # theoretical best case (actual - travel)
    return (total, control_total)

def gpu_test(frames, name, fn, ctrl, img):
    gpu_run_effect(fn, img)
    start = timer()
    for _ in range(frames):
        result = gpu_run_effect(fn, img)
    total = timer() - start

    gpu_run_effect(ctrl, img)
    start = timer()
    for _ in range(frames):
        result = gpu_run_effect(ctrl, img)
    control_total = timer() - start

    print(name + ' gpu-test')
    print('%f fps' % (frames / total)) # actual time
    print('%f fps' % (frames / control_total)) # travel time
    print('%f fps' % (frames / (total-control_total))) # theoretical best case (actual - travel)
    return (total, control_total)


def build_control(img):
    control_template = Template("""
    __global__ void control(unsigned char *dest, unsigned char *img)
    {
        const int row = threadIdx.x + blockDim.x*blockIdx.x;
        const int col = threadIdx.y + blockDim.y*blockIdx.y;
        const int chan = threadIdx.z;

        const int idx = chan + col*{{depth}} + row*{{depth}}*{{height}};
        if (idx > {{width}}*{{height}}*{{depth}}) {
            return;
        }
        dest[idx] = img[idx];
    }
    """)
    width, height, depth = img.shape
    module = SourceModule(control_template.render(width=width, height=height, depth=depth))

    gpu_brighten = module.get_function("control")
    block = (8,8,3)
    grid = (ceil(width/8), ceil(height/8))

    gpu_control = (gpu_brighten, block, grid)

    def vectorized_control(px):
        return px

    cpu_control = numpy.vectorize(vectorized_control)

    return (cpu_control, gpu_control)


frames = 100
results = list()
name = chosen_effect + ' ' + chosen_file
button = Button(root, text='run perf test', command=lambda: perf_test(frames, name,cpu_renderer, gpu_renderer, img, root, results))
button.pack(side=BOTTOM)
root.mainloop()

if len(results) == 0:
    exit()

def fps_calc(total, control_total):
    fps = frames/total
    control_fps = frames/control_total
    best_fps = frames/(total-control_total)

    return (fps, control_fps, best_fps)

cpu_fps, cpu_control_fps, cpu_best_fps = fps_calc(results[0], results[1])
gpu_fps, gpu_control_fps, gpu_best_fps = fps_calc(results[2], results[3])


fps_stats = [cpu_fps, cpu_control_fps, gpu_fps, gpu_control_fps]


def fps_formatter(x, pos):
    return '%1.0f FPS' % x


formatter = FuncFormatter(fps_formatter)
x = numpy.arange(4)


_, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
ax = axs[0]
ax.yaxis.set_major_formatter(formatter)
ax.set_ylabel('FPS linear scale')
ax.bar(x, fps_stats)
plt.xticks(x, ('CPU', 'CPU CTRL', 'GPU', 'GPU CTRL'))

ax = axs[1]
ax.set_yscale("log", nonposy='clip')
ax.set_ylabel('FPS logarithmic scale')
ax.bar(x, fps_stats)

fig = plt.gcf()
fig.canvas.set_window_title(name)
plt.tight_layout()
plt.show()

