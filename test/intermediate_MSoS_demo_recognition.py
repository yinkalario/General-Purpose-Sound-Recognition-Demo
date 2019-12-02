#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:57:07 2019

@author: ck0022
"""

import sys
from time import sleep
import tkinter as tk
from tkinter import ttk as ttk
from tkinter import font
from threading import Thread


class DemonstratorGUI(ttk.Frame):
    
    def __init__(self, master=None,number_of_results=None):
        super().__init__(master)
        self.master = master
        
        self.banner = tk.PhotoImage(file="./images/msos_banner.ppm") 
        self.logo = [tk.PhotoImage(file="./images/msos_logo_low_res.ppm"),
                     tk.PhotoImage(file="./images/Surrey_Logo_Blue_Gold.ppm"),
                     tk.PhotoImage(file="./images/EPSRC-lowres.ppm"),
                     tk.PhotoImage(file="./images/sup_logo_ec.ppm")]
        
        self.demoStyle = ttk.Style()
        self.numberOfResults = int(number_of_results)
        
        self.posthocProb = [None] * self.numberOfResults
        for k in range(self.numberOfResults):
            self.posthocProb[k] = tk.IntVar()
            self.posthocProb[k].set(1)
        
        self.resultLabel = [None] * self.numberOfResults
        for q in range(self.numberOfResults):
            self.resultLabel[q] = tk.StringVar()
            tempPHStr = 'AAA'*(q+1)
            placeholderStr='{0:<12.12}'.format(tempPHStr)
            self.resultLabel[q].set(placeholderStr)
                
       # demoStyle.configure('Large.TFrame',height=400,width=600)
        self.demoStyle.configure('Large.TFrame', height=400, width=600, background='white')
        self['style'] = 'Large.TFrame'
        
        self.pack()
        #print(self.winfo_class())
        self.create_widgets()
        
        

    def create_widgets(self):
        
        self.procStatus = False
        
        #demoStyle.theme_use('aqua')
        
        # ------------------ General settings -----------------------
        titleFont = font.Font(family='Helvetica', size=24, weight='bold')
        buttonFont = font.Font(family='Helvetica', size=16)
        labelHeadFont = font.Font(family='Helvetica', size=18)
        labelEntryFont = font.Font(family='Helvetica', size=14,weight='bold')
        labelEntryLabFont = font.Font(family='Courier', size=14)
        
        self.demoStyle.configure('General.TButton', foreground='black', background='white',
                            font=buttonFont)
        
        
        # ------------------ Banner -------------------------
        self.demoStyle.configure('banner.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(self,image = self.banner,style='banner.TLabel')
        titleLabel.pack(side='top', padx=0, pady=0)
        
        # ------------------ Title -------------------------
        self.demoStyle.configure('title.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(self,text='Sound Recognition',style='title.TLabel')
        titleLabel.pack(side='top', padx=20, pady=20)

        # ------------------ Start & stop button -----------
        self.demoStyle.configure('StartStop.TFrame', background='white')
        startStopFrame = ttk.Frame(self,style='StartStop.TFrame')
        startStopFrame.pack(side='top', padx=10, pady=10)        
        
        self.demoStyle.configure('StartStop.TButton', foreground='black', background='white', 
                            font=buttonFont)
        self.demoStyle.map("StartStop.TButton",background=[('active', '#7cb669')])
        
        self.startButton = ttk.Button(startStopFrame,text='Start',style='StartStop.TButton')
        self.startButton['command'] = self.start_stop_recording_and_analysis
        self.startButton.pack(side='left', padx=30, pady=10, ipadx=80, ipady=20)
        

        # ------------------ Results master frame -----------
        self.demoStyle.configure('ResultsOverall.TFrame', background='white')
        resultsOverallFrame = ttk.Frame(self,style='ResultsOverall.TFrame')
        resultsOverallFrame.pack(padx=20, pady=0)
        
        
        # ------------------ Results headings -----------
        self.demoStyle.configure('headResults.TLabel', background='white',font=labelHeadFont)
        
        self.demoStyle.configure('ResultsHeading.TFrame', background='white')
        resultsHeadingFrame = ttk.Frame(resultsOverallFrame,style='ResultsHeading.TFrame')
        resultsHeadingFrame.pack(side='top', padx=20, pady=10)
        
        resRankLabel = ttk.Label(resultsHeadingFrame,text='Rank',style='headResults.TLabel')
        resRankLabel.pack(side='left',padx=40, pady=20)
         
        resLabLabel = ttk.Label(resultsHeadingFrame,text='{0:<12}'.format('Label'),style='headResults.TLabel')
        resLabLabel.pack(side='left',padx=10, pady=20)
    
        resConfLabel = ttk.Label(resultsHeadingFrame,text='Confidence',style='headResults.TLabel')
        resConfLabel.pack(side='left',padx=40, pady=20)
        
        
        # ------------------ Results entries -----------
        self.demoStyle.configure('entryResults.TLabel',background='white',font=labelEntryFont)
        self.demoStyle.configure('entryLabResults.TLabel',background='white',font=labelEntryLabFont)
        self.demoStyle.configure("entryConf.Horizontal.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        
        resultsRowFrame = [None] * self.numberOfResults
        self.resRankEntryLabel = [None] * self.numberOfResults
        self.resLabelEntryLabel = [None] * self.numberOfResults
        self.resConfEntryLabel = [None] * self.numberOfResults              
        for n in range(self.numberOfResults):
            
            # Row frame
            self.demoStyle.configure('ResultsRow.TFrame', background='white')
            resultsRowFrame[n] = ttk.Frame(resultsOverallFrame,style='ResultsRow.TFrame')
            resultsRowFrame[n].pack(side='top', padx=20, pady=10)
            
            # Rank label
            self.resRankEntryLabel[n] = ttk.Label(resultsRowFrame[n],text='{0}'.format(n+1),style='entryResults.TLabel')
            self.resRankEntryLabel[n].pack(side='left', padx=10, pady=5)
        
            # Labels of recognised sounds
            #placeholderStr = 'AAA'*(n+1)
            self.resLabelEntryLabel[n] = ttk.Label(resultsRowFrame[n],style='entryLabResults.TLabel',
                                   textvariable=self.resultLabel[n],)
            self.resLabelEntryLabel[n].pack(side='left', padx=10, pady=5)
            
            # Confidence bars
            self.resConfEntryLabel[n] = ttk.Progressbar(resultsRowFrame[n],style="entryConf.Horizontal.TProgressbar",
                                  orient="horizontal",length=200,mode="determinate",maximum=1,
                                  variable=self.posthocProb[n]) 
            self.resConfEntryLabel[n].pack(side='left', padx=10, pady=5)
        
           # value=(10-n)*0.1
        
        # ------------------ Exit button -----------
        self.demoStyle.configure('Exit.TButton', foreground='#7d150e', background='white', font=buttonFont)
        self.exitButton = ttk.Button(self,text='Exit', style='Exit.TButton')
        self.exitButton['command'] = self.exit_demo
        self.exitButton.pack(padx=20, pady=40, ipadx=80, ipady=20)
        
        # ------------------ Logos -------------------------
        
        self.demoStyle.configure('logos.TFrame', background='white')
        logosFrame = ttk.Frame(self,style='logos.TFrame')
        logosFrame.pack(side='bottom', padx=10, pady=5)
        
        self.demoStyle.configure('logos.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(logosFrame,image = self.logo[1],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        self.demoStyle.configure('logos.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(logosFrame,image = self.logo[0],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        self.demoStyle.configure('logos.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(logosFrame,image = self.logo[2],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        self.demoStyle.configure('logos.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(logosFrame,image = self.logo[3],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)



    def start_stop_recording_and_analysis(self):
        if self.procStatus:
            self.startButton['text'] = 'Start'
            self.demoStyle.configure('StartStop.TButton', background='#7cb669')
            self.demoStyle.map("StartStop.TButton",background=[('active', '#689958')])
            self.procStatus = False    
        else:
            self.startButton['text'] = 'Stop'
            self.demoStyle.configure('StartStop.TButton', background='#b68169')
            self.demoStyle.map("StartStop.TButton",background=[('active', '#996c58')])
            self.procStatus = True
            
            self.threadAnalysis = Thread(target=self.run_analysis)
            self.threadAnalysis.start()
            
            
    def run_analysis(self):
        
        # ------------  TEMPORARY FOR TESTING GUI ------------------
        tempLabels = ['AAA','BBB']
        nnn = 1
        while self.procStatus:
            v = (nnn + 1) / 10
            self.posthocProb[0].set(v)
            
            if nnn == 5:
                tempStr='{0:<12.12}'.format(tempLabels[0])
                self.resultLabel[0].set(tempStr)
            elif nnn == 10:
                tempStr='{0:<12.12}'.format(tempLabels[1])
                self.resultLabel[0].set(tempStr)
            
            sleep(0.1)
            nnn = nnn%10 + 1 
        # ------------  TEMPORARY FOR TESTING GUI ------------------    
            
            
    def exit_demo(self):
        if self.procStatus:
            self.start_stop_recording_and_analysis()        
        self.master.destroy()
        

    
def main(numberOfResults=None): 
     
    if numberOfResults is None and len(sys.argv) > 1:
        numberOfResults = sys.argv[1]
    else: 
        numberOfResults = 5 
    
    root = tk.Tk()
    demo = DemonstratorGUI(master=root,number_of_results=numberOfResults)
    demo.master.title('Making Sense of Sounds - Sound Recognition')
    demo.mainloop()
    
   
if __name__ == '__main__':
    main() 
    