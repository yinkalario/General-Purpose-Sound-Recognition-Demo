#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import tkinter as tk
from tkinter import ttk as ttk

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
s = ttk.Style()
#s.theme_use('clam')
s.configure("red.Horizontal.TProgressbar", foreground='red', background='red')
tt = ttk.Progressbar(frame, style="red.Horizontal.TProgressbar", orient="horizontal", length=300, mode="determinate", maximum=4, value=1)
tt.pack()

