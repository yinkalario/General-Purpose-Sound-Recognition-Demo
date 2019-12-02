#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:49:35 2019

@author: ck0022
"""

from tkinter import *

root = Tk()
frame = Frame(root,bg="white")
frame.pack(pady=200)

bottomframe = Frame(root,bg="white")
bottomframe.pack(side = BOTTOM, padx=200)

redbutton = Button(frame, text="Red", fg="red")
redbutton.pack( side = LEFT)

greenbutton = Button(frame, text="Brown", fg="brown")
greenbutton.pack( side = LEFT )

bluebutton = Button(frame, text="Blue", fg="blue")
bluebutton.pack( side = LEFT )

blackbutton = Button(bottomframe, text="Black", fg="black")
blackbutton.pack( side = BOTTOM)

root.mainloop()