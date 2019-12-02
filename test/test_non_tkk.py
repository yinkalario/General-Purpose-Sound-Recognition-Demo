#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:57:07 2019

@author: ck0022
"""

import tkinter as tk

class DemonstratorGUI(ttk.Frame):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        

        self.pack()
        #print(self.winfo_class())
        self.create_widgets()

    def create_widgets(self):
        
        
        buttonFrame = tk.Frame(self, bg="white")
        buttonFrame.pack(side='top', padx=100, pady=50)
        
        self.startButton = tk.Button(buttonFrame,text='Start',fg="green", bg="white")
        self.startButton["command"] = self.show_result
        self.startButton.pack(side="left", pady=20, padx=20)
        
        self.stopButton = tk.Button(buttonFrame,text="Stop",fg="blue", bg="white")
        self.stopButton["command"] = self.show_result
        self.stopButton.pack(side="top", pady=20, padx=20)

        resultsFrame = tk.Frame(self,bg="yellow")
        resultsFrame.pack(padx=100, pady=100)
              
        self.exitButton = tk.Button(self,text="Exit",fg="red", bg="white",
                              command=self.master.destroy)
        self.exitButton.pack(side="bottom",pady=50)
        
        
        
        

    def show_result(self):
        print("This was a hand clap")

root = tk.Tk()
demo = DemonstratorGUI(master=root)
demo.master.title("Sound Recognition")
demo.master.color="white"
demo.mainloop()