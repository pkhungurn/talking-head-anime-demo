import os
import sys

sys.path.append(os.getcwd())

from tkinter import Frame, Label, BOTH, Tk, LEFT, HORIZONTAL, Scale, Button, GROOVE, filedialog, PhotoImage, messagebox

import PIL.Image
import PIL.ImageTk
import numpy
import torch

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from poser.poser import Poser
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import extract_pytorch_image_from_filelike, rgba_to_numpy_image


class ManualPoserApp:
    def __init__(self,
                 master,
                 poser: Poser,
                 torch_device: torch.device):
        super().__init__()
        self.master = master
        self.poser = poser
        self.torch_device = torch_device

        self.master.title("Manual Poser")

        source_image_frame = Frame(self.master, width=256, height=256)
        source_image_frame.pack_propagate(0)
        source_image_frame.pack(side=LEFT)

        self.source_image_label = Label(source_image_frame, text="Nothing yet!")
        self.source_image_label.pack(fill=BOTH, expand=True)

        control_frame = Frame(self.master, borderwidth=2, relief=GROOVE)
        control_frame.pack(side=LEFT, fill='y')

        self.param_sliders = []
        for param in self.poser.pose_parameters():
            slider = Scale(control_frame,
                           from_=param.lower_bound,
                           to=param.upper_bound,
                           length=256,
                           resolution=0.001,
                           orient=HORIZONTAL)
            slider.set(param.default_value)
            slider.pack(fill='x')
            self.param_sliders.append(slider)

            label = Label(control_frame, text=param.display_name)
            label.pack()

        posed_image_frame = Frame(self.master, width=256, height=256)
        posed_image_frame.pack_propagate(0)
        posed_image_frame.pack(side=LEFT)

        self.posed_image_label = Label(posed_image_frame, text="Nothing yet!")
        self.posed_image_label.pack(fill=BOTH, expand=True)

        self.load_source_image_button = Button(control_frame, text="Load Image ...", relief=GROOVE,
                                               command=self.load_image)
        self.load_source_image_button.pack(fill='x')

        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)

    def load_image(self):
        file_name = filedialog.askopenfilename(
            filetypes=[("PNG", '*.png')],
            initialdir="data/illust")
        if len(file_name) > 0:
            image = PhotoImage(file=file_name)
            if image.width() != self.poser.image_size() or image.height() != self.poser.image_size():
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.width(), image.height(), self.poser.image_size(), self.poser.image_size())
                messagebox.showerror("Wrong image size!", message)
            self.source_image_label.configure(image=image, text="")
            self.source_image_label.image = image
            self.source_image_label.pack()

            self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)
            self.needs_update = True

    def update_pose(self):
        self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
        for i in range(self.pose_size):
            self.current_pose[i] = self.param_sliders[i].get()
        self.current_pose = self.current_pose.unsqueeze(dim=0)

    def update_image(self):
        self.update_pose()
        if (not self.needs_update) and self.last_pose is not None and (
                (self.last_pose - self.current_pose).abs().sum().item() < 1e-5):
            self.master.after(1000 // 30, self.update_image)
            return
        if self.source_image is None:
            self.master.after(1000 // 30, self.update_image)
            return
        self.last_pose = self.current_pose

        posed_image = self.poser.pose(self.source_image, self.current_pose).detach().cpu()
        numpy_image = rgba_to_numpy_image(posed_image[0])
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        photo_image = PIL.ImageTk.PhotoImage(image=pil_image)

        self.posed_image_label.configure(image=photo_image, text="")
        self.posed_image_label.image = photo_image
        self.posed_image_label.pack()
        self.needs_update = False

        self.master.after(1000 // 30, self.update_image)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)
    root = Tk()
    app = ManualPoserApp(master=root, poser=poser, torch_device=cuda)
    root.mainloop()
