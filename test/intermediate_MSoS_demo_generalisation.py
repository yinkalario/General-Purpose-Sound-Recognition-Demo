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
from tkinter import filedialog
from threading import Thread


class DemonstratorGUI(ttk.Frame):
    
    def __init__(self, master=None,recgn_classes=None,default_sound_data_path=None):
        super().__init__(master)
        self.master = master
        
        self.defaultSoundDataPath = default_sound_data_path
        
        self.banner = tk.PhotoImage(file='./images/msos_banner.ppm') 
        self.logo = [tk.PhotoImage(file='./images/msos_logo_low_res.ppm'),
                     tk.PhotoImage(file='./images/Surrey_Logo_Blue_Gold.ppm'),
                     tk.PhotoImage(file='./images/EPSRC-lowres.ppm'),
                     tk.PhotoImage(file='./images/sup_logo_ec.ppm'),
                     tk.PhotoImage(file='./images/CVSSP logo_MK_7Feb2019_v2_black.ppm')]
        
        self.demoStyle = ttk.Style()
        self.recognisedClasses = recgn_classes
        self.numberOfClasses = len(self.recognisedClasses)
        
        self.soundFilename = None
        
        self.recording = False
        self.soundData = None
        
        self.posthocProb = [None] * len(self.recognisedClasses)
        for k in range(self.numberOfClasses):
            self.posthocProb[k] = tk.IntVar()
            self.posthocProb[k].set(1/(k+1))
        
        self.resultRank = [None] * len(self.recognisedClasses)
        for q in range(self.numberOfClasses):
            self.resultRank[q] = tk.StringVar()
            self.resultRank[q].set('-')
                
        self.demoStyle.configure('Large.TFrame', height=400, width=600, background='white')
        self['style'] = 'Large.TFrame'
        
        self.pack()

        self.create_widgets()
        
        

    def create_widgets(self):
        
        
        # ------------------ General settings -----------------------
        titleFont = font.Font(family='Helvetica', size=24, weight='bold')
        selectSoundFont = font.Font(family='Helvetica', size=14)
        selectionFont = font.Font(family='Courier', size=12)
        buttonFont = font.Font(family='Helvetica', size=14)
        labelEntryFont = font.Font(family='Helvetica', size=16)
        labelEntryRankFont = font.Font(family='Courier', size=18,weight='bold')
        
        self.demoStyle.configure('General.TButton', foreground='black', background='white',
                            font=buttonFont)
        
        
        # ------------------ Banner -------------------------
        self.demoStyle.configure('banner.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(self,image = self.banner,style='banner.TLabel')
        titleLabel.pack(side='top', padx=0, pady=0)
        
        # ------------------ Title -------------------------
        self.demoStyle.configure('title.TLabel', background='white',font=titleFont)
        titleLabel = ttk.Label(self,text='Human-like Sound Categorisation',style='title.TLabel')
        titleLabel.pack(side='top', padx=20, pady=20)



        # ------------------ Acquisition buttons -----------
        self.demoStyle.configure('acquisition.TFrame', background='white')
        acquisitionFrame = ttk.Frame(self,style='acquisition.TFrame')
        acquisitionFrame.pack(side='top', padx=10, pady=0)        
        
        self.demoStyle.configure('load.TButton', foreground='black', background='white', 
                            font=buttonFont)
        self.demoStyle.map("load.TButton",background=[('active', '#7cb669')])
        
        self.loadButton = ttk.Button(acquisitionFrame,text=' Load ',style='load.TButton')
        self.loadButton['command'] = self.load_sound
        self.loadButton.pack(side='left', padx=20, pady=5, ipadx=10, ipady=10)
        
        
        self.demoStyle.configure('recordOff.TButton', foreground='black', background='white', 
                            font=buttonFont)
        self.demoStyle.map("recordOff.TButton",background=[('active', '#7cb669')])
        
        self.demoStyle.configure('recordOn.TButton', foreground='black', background='#b68169', 
                            font=buttonFont)
        self.demoStyle.map("recordOn.TButton",background=[('active', '#b68169')])
        
        self.recordButton = ttk.Button(acquisitionFrame,text='Record',style='recordOff.TButton')
        self.recordButton['command'] = self.record_sound
        self.recordButton.pack(side='left', padx=20, pady=5, ipadx=10, ipady=10)
        
        self.demoStyle.configure('play.TButton', foreground='black', background='white', 
                            font=buttonFont)
        self.demoStyle.map("play.TButton",background=[('active', '#7cb669')])
        
        self.playButton = ttk.Button(acquisitionFrame,text=' Play ',style='load.TButton')
        self.playButton['command'] = self.play_sound
        self.playButton.pack(side='left', padx=20, pady=5, ipadx=10, ipady=10)
        

        # ------------------ Selection -------------------------
        self.demoStyle.configure('selection.TFrame', background='white')
        selectionFrame = ttk.Frame(self,style='selection.TFrame')
        selectionFrame.pack(side='top', padx=10, pady=10) 
        
        self.demoStyle.configure('selectSound.TLabel', background='white',font=selectSoundFont)
        selectSoundLabel = ttk.Label(selectionFrame,text='Selected sound:',style='selectSound.TLabel')
        selectSoundLabel.pack(side='left', padx=10, pady=10)

        self.demoStyle.configure('selection.TLabel', background='white',font=selectionFont)
        selectionLabel = ttk.Label(selectionFrame,text='-----------',style='selection.TLabel')
        selectionLabel.pack(side='left', padx=10, pady=10)



        # ------------------ Analysis button -----------
        self.demoStyle.configure('analysis.TFrame', background='white')
        acquisitionFrame = ttk.Frame(self,style='analysis.TFrame')
        acquisitionFrame.pack(side='top', padx=10, pady=0)        
        
        self.demoStyle.configure('analysis.TButton', foreground='black', background='white', 
                            font=buttonFont)
        self.demoStyle.map("analysis.TButton",background=[('active', '#7cb669')])
        
        self.loadButton = ttk.Button(acquisitionFrame,text='Analyse',style='analysis.TButton')
        self.loadButton['command'] = self.run_analysis
        self.loadButton.pack(padx=20, pady=10, ipadx=80, ipady=10)
        



        # ------------------ Results master frame -----------
        self.demoStyle.configure('ResultsOverall.TFrame', background='white')
        resultsOverallFrame = ttk.Frame(self,style='ResultsOverall.TFrame')
        resultsOverallFrame.pack(padx=20, pady=10)
        
        
        
        
        # ------------------ Results entries -----------
        self.demoStyle.configure('entryResults.TLabel',background='white',font=labelEntryFont)
        self.demoStyle.configure('entryResultsBest.TLabel',background='white',foreground='#1d6a0b',font=labelEntryFont)
        self.demoStyle.configure('entryResultsRank.TLabel',background='white',font=labelEntryRankFont)
        self.demoStyle.configure("entryConf.Vertical.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        
        
        resultsColumnFrame = [None] * self.numberOfClasses
        self.resLabelEntryLabel = [None] * self.numberOfClasses
        self.resRankEntryLabel = [None] * self.numberOfClasses
        self.resConfEntryLabel = [None] * self.numberOfClasses              
        for n in range(self.numberOfClasses):
            
            # Row frame
            self.demoStyle.configure('ResultsColumn.TFrame', background='white')
            resultsColumnFrame[n] = ttk.Frame(resultsOverallFrame,style='ResultsColumn.TFrame')
            resultsColumnFrame[n].pack(side='left', padx=10, pady=10)
            
        
            # Labels of recognised categories
            self.resLabelEntryLabel[n] = ttk.Label(resultsColumnFrame[n],style='entryResults.TLabel',
                                   text=self.recognisedClasses[n])
                
                
            self.resLabelEntryLabel[n].pack(side='top', padx=5, pady=10)
            
            # Rank label
            self.resRankEntryLabel[n] = ttk.Label(resultsColumnFrame[n],textvariable=self.resultRank[n],style='entryResultsRank.TLabel')
            self.resRankEntryLabel[n].pack(side='top', padx=5, pady=10)
            
            # Confidence bars
            self.resConfEntryLabel[n] = ttk.Progressbar(resultsColumnFrame[n],style="entryConf.Vertical.TProgressbar",
                                  orient="vertical",length=150,mode="determinate",maximum=1,
                                  variable=self.posthocProb[n]) 
            self.resConfEntryLabel[n].pack(side='bottom', padx=5, pady=10)
        
           
        
        # ------------------ Exit button -----------
        self.demoStyle.configure('Exit.TButton', foreground='#7d150e', background='white', font=buttonFont)
        self.exitButton = ttk.Button(self,text='Exit', style='Exit.TButton')
        self.exitButton['command'] = self.exit_demo
        self.exitButton.pack(padx=20, pady=20, ipadx=80, ipady=10)
        
        # ------------------ Logos -------------------------
        
        self.demoStyle.configure('logos.TFrame', background='white')
        logosFrame = ttk.Frame(self,style='logos.TFrame')
        logosFrame.pack(side='bottom', padx=10, pady=20)
        
        self.demoStyle.configure('logos.TLabel', background='white')
              
        titleLabel = ttk.Label(logosFrame,image = self.logo[1],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        titleLabel = ttk.Label(logosFrame,image = self.logo[4],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        titleLabel = ttk.Label(logosFrame,image = self.logo[0],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        titleLabel = ttk.Label(logosFrame,image = self.logo[2],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)

        titleLabel = ttk.Label(logosFrame,image = self.logo[3],style='logos.TLabel')
        titleLabel.pack(side='left', padx=10, pady=0)



    def load_sound(self):
        if not(self.recording):
            
            self.soundFilename = filedialog.askopenfilename(initialdir = self.defaultSoundDataPath,
                                    title = "Select file", filetypes = (("WAV files","*.wav"),("All files","*.*")))
            
            if not(self.soundFilename == ''):  # if not canceled by user
               
                  # *** call loading function, store data in self.soundData 
                  # and parameters needed for playing in appropriately named other ***
                  # fields of self 
                  
                  # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------
                  print (self.soundFilename)
                  # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------
        else:
            print('Busy')
            pass
        
        
    def record_sound(self):
        if not(self.recording):
            
            self.recordButton.configure(style='recordOn.TButton',text='Recording')
            self.recordButton.update()
            
            self.recording = True 
            
            # *** call recording function, store data in self.soundData 
            # and parameters needed for playing in appropriately named other 
            # fields of self ***
            
            # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------
            print('Start')
            sleep(5)
            print('Stop')
            # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------

                 
            self.recording = False
            self.recordButton.configure(style='recordOff.TButton',text='Record')
            self.soundFilename = None
            
        
        
    def play_sound(self):
        if not(self.recording):
            print('PLaying')
           
            # Play sound (will need some parameters, e.g, sample rate)
            # Most sound players play the sound in the bckground and hand
            # back control to the calling routine. Thus, GUI does not need to
            # be blocked and user can continue with e.g. the analysing function
            
            
            
    def run_analysis(self):
        
        if not(self.recording):
            
            # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------
            
            # Demonstrates the ranking display - order of classes (= indices) according to 
            # variable RECOGNISED_CLASSES in main
            self.resultRank[0].set('4')
            self.resultRank[1].set('1')
            self.resultRank[2].set('3')
            self.resultRank[3].set('5')
            self.resultRank[4].set('2')
            
            # Demonstrates the posthoc prob display - order of classes (= indices) according to 
            # variable RECOGNISED_CLASSES in main
            self.posthocProb[0].set(0.2)
            self.posthocProb[1].set(0.9)
            self.posthocProb[2].set(0.34)
            self.posthocProb[3].set(0.103)
            self.posthocProb[4].set(0.44)
            
            # demonstrates how the label colour of the best class is changed
            # here the one with index 1 ('Music')
            self.resLabelEntryLabel[1].configure(style='entryResultsBest.TLabel')
            
             # ------------  TEMPORARY FOR TESTING GUI * REPLACE! * ------------------    
         
            
            
    def exit_demo(self):
        
        self.master.destroy()
        

    
def main(defaultSoundDataPath=None): 
     
    RECOGNISED_CLASSES = ['Nature', 'Music', 'Human', 'Effects', 'Urban']
    
    if defaultSoundDataPath is None and len(sys.argv) > 1:
        defaultSoundDataPath = sys.argv[1]
    else: 
        defaultSoundDataPath = '.' # Would that work under MS Windows,too?
    
    root = tk.Tk()
    demo = DemonstratorGUI(master=root,recgn_classes=RECOGNISED_CLASSES,default_sound_data_path=defaultSoundDataPath)
    demo.master.title('Making Sense of Sounds - Categorise sounds like a human')
    demo.mainloop()
    
   
if __name__ == '__main__':
    main() 
    