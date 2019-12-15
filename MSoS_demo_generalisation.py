#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:57:07 2019

@author: ck0022
"""

import sys
import tkinter as tk
from datetime import datetime
from os import path
from tkinter import filedialog, font
from tkinter import ttk as ttk

import numpy as np

import audio_analysis


class DemonstratorGUI(ttk.Frame):
    
    def __init__(self, master=None,recgn_classes=None,default_sound_data_path=None):
        
        super().__init__(master)
        self.master = master
        
        self.masterFrameSwitchWidth = 560
        self.masterFrameSwitchHeight = 840
        
        if self.winfo_screenheight() < 900 or self.winfo_screenwidth() < 600 :
            startUpSize = 'small'
        else:
            startUpSize = 'normal'
        
    
        self.defaultSoundDataPath = default_sound_data_path
        
        self.banner = tk.PhotoImage(file='./images/msos_banner.ppm') 
        self.bannerSmall = tk.PhotoImage(file='./images/msos_banner_small.ppm') 
        self.logo = [tk.PhotoImage(file='./images/Surrey_Logo_Blue_Gold.ppm'),
                     tk.PhotoImage(file='./images/CVSSP logo_MK_7Feb2019_v2_black.ppm'),
                     tk.PhotoImage(file='./images/msos_logo_low_res.ppm'),     
                     tk.PhotoImage(file='./images/EPSRC-lowres.ppm'),
                     tk.PhotoImage(file='./images/sup_logo_ec.ppm')
                     ]
        self.logoSmall = [tk.PhotoImage(file='./images/Surrey_Logo_Blue_Gold_small.ppm'),
                          tk.PhotoImage(file='./images/CVSSP logo_MK_7Feb2019_v2_black_small.ppm'),
                          tk.PhotoImage(file='./images/msos_logo_low_res_small.ppm'),     
                          tk.PhotoImage(file='./images/EPSRC-lowres_small.ppm'),
                          tk.PhotoImage(file='./images/sup_logo_ec_small.ppm')
                          ]
        
        self.demoStyle = ttk.Style()
        self.font = None
        
        self.recognisedClasses = recgn_classes
        self.numberOfClasses = len(self.recognisedClasses)
        
        self.numberOneInRanking = 0;
    
        self.soundFilename = None
        
        self.soundName = tk.StringVar()
        self.soundName.set('-----------')
        
        self.recording = False
       
        self.audioAnalysis = audio_analysis.AudioAnalysis()
        
        
        self.posthocProb = [None] * len(self.recognisedClasses)
        for k in range(self.numberOfClasses):
            self.posthocProb[k] = tk.IntVar()
            self.posthocProb[k].set(0)
        
        self.resultRank = [None] * len(self.recognisedClasses)
        for q in range(self.numberOfClasses):
            self.resultRank[q] = tk.StringVar()
            self.resultRank[q].set('-')
            
        self.define_styles()    
        self['style'] = 'ourMaster.TFrame'      
        self.pack(fill="both",expand=True)
        
        self.create_widgets()
        
        self.overallSize = startUpSize
        self.image_selection(startUpSize)
        self.apply_styles(startUpSize)
        self.apply_packing(startUpSize)
        
        self.bind("<Configure>", self.track_overall_size)
        

    def create_widgets(self):
        
        # ------------------ Banner -------------------------
        self.bannerFrame = ttk.Frame(self)
        self.bannerLabel = ttk.Label(self.bannerFrame)
              
        
        # ------------------ Title -------------------------
        self.titleFrame = ttk.Frame(self,style='title.TFrame')
        self.titleLabel = ttk.Label(self.titleFrame,text='Human-like Sound Categorisation')


        # ------------------ Acquisition buttons -----------
        self.acquisitionFrame = ttk.Frame(self)
        
        self.loadButton = ttk.Button(self.acquisitionFrame,text=' Load ')
        self.loadButton['command'] = self.load_sound
                       
        self.recordButton = ttk.Button(self.acquisitionFrame,text='Record')
        self.recordButton['command'] = self.record_sound
        
        self.playButton = ttk.Button(self.acquisitionFrame,text=' Play ')
        self.playButton['command'] = self.play_sound
        

        # ------------------ Selection -------------------------
        self.selectionFrame = ttk.Frame(self)
        self.selectSoundLabel = ttk.Label(self.selectionFrame,text='Selected sound:')
        self.selectionLabel = ttk.Label(self.selectionFrame,textvariable=self.soundName)
        

        # ------------------ Analysis button -----------
        self.analysisFrame = ttk.Frame(self)
        self.analysisButton = ttk.Button(self.analysisFrame,text='Analyse')
        self.analysisButton['command'] = self.run_analysis
        

        # ------------------ Results master frame -----------
        self.resultsOverallFrame = ttk.Frame(self)
    
        
        # ------------------ Results entries -----------
        
        self.resultsColumnFrame = [None] * self.numberOfClasses
        self.resLabelEntryLabel = [None] * self.numberOfClasses
        self.resRankEntryLabel = [None] * self.numberOfClasses
        self.resConfEntryLabel = [None] * self.numberOfClasses              
        for n in range(self.numberOfClasses):
            
            # Row frame
            self.resultsColumnFrame[n] = ttk.Frame(self.resultsOverallFrame)
        
            # Labels of recognised categories
            self.resLabelEntryLabel[n] = ttk.Label(self.resultsColumnFrame[n],text=self.recognisedClasses[n])
                    
            # Rank label
            self.resRankEntryLabel[n] = ttk.Label(self.resultsColumnFrame[n],textvariable=self.resultRank[n])
               
            # Confidence bars
            self.resConfEntryLabel[n] = ttk.Progressbar(self.resultsColumnFrame[n],
                                  orient="vertical",length=150,mode="determinate",maximum=1,
                                  variable=self.posthocProb[n]) 
        
        
        # ------------------ Exit button -----------
        self.exitFrame = ttk.Frame(self)
        
        self.exitButton = ttk.Button(self.exitFrame,text='Exit')
        self.exitButton['command'] = self.exit_demo
        
        
        # ------------------ Logos -------------------------
        self.logosFrame = ttk.Frame(self)       
        self.logoLabel = [None] * len(self.logo)
        for l in range(len(self.logo)):  
            self.logoLabel[l] = ttk.Label(self.logosFrame,image = self.logo[l])
            
 
        
        
    def define_styles(self):
        
        self.font = self.ourFonts()
        
        # Frames
        self.demoStyle.configure('ourMaster.TFrame', background='white')
        self.demoStyle.configure('banner.TFrame', background='white', border='none')
        self.demoStyle.configure('title.TFrame', background='white')
        self.demoStyle.configure('acquisition.TFrame', background='white')
        self.demoStyle.configure('selection.TFrame', background='white')
        self.demoStyle.configure('analysis.TFrame', background='white')
        self.demoStyle.configure('resultsOverall.TFrame', background='white')
        self.demoStyle.configure('resultsColumn.TFrame', background='white')
        self.demoStyle.configure('exit.TFrame', background='white')
        self.demoStyle.configure('logos.TFrame', background='white')
        
        # Labels
        self.demoStyle.configure('banner.TLabel', background='white',font=self.font.title)
        self.demoStyle.configure('title.TLabel', background='white',font=self.font.title)
        self.demoStyle.configure('titleSmall.TLabel', background='white',font=self.font.titleSmall)
        self.demoStyle.configure('selectSound.TLabel', background='white',font=self.font.selectSound)
        self.demoStyle.configure('selectSoundSmall.TLabel', background='white',font=self.font.selectSoundSmall)
        self.demoStyle.configure('selection.TLabel', background='white',font=self.font.selection)
        self.demoStyle.configure('selectionSmall.TLabel', background='white',font=self.font.selectionSmall)
        self.demoStyle.configure('entryResults.TLabel',background='white',font=self.font.labelEntry)
        self.demoStyle.configure('entryResultsSmall.TLabel',background='white',font=self.font.labelEntrySmall)
        self.demoStyle.configure('entryResultsBest.TLabel',background='white',foreground='#1d6a0b',font=self.font.labelEntry)
        self.demoStyle.configure('entryResultsBestSmall.TLabel',background='white',foreground='#1d6a0b',font=self.font.labelEntrySmall)
        self.demoStyle.configure('entryResultsRank.TLabel',background='white',font=self.font.labelEntryRank)
        self.demoStyle.configure('entryResultsRankSmall.TLabel',background='white',font=self.font.labelEntryRankSmall)
        self.demoStyle.configure('logos.TLabel', background='white')
        
        # Buttons
        self.demoStyle.configure('load.TButton', foreground='black', background='white', font=self.font.button)
        self.demoStyle.map("load.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('loadSmall.TButton', foreground='black', background='white', font=self.font.buttonSmall)
        self.demoStyle.map("loadSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('recordOff.TButton', foreground='black', background='white', font=self.font.button)
        self.demoStyle.map("recordOff.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('recordOffSmall.TButton', foreground='black', background='white', font=self.font.buttonSmall)
        self.demoStyle.map("recordOffSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('recordOn.TButton', foreground='black', background='#b68169',font=self.font.button)
        self.demoStyle.map("recordOn.TButton",background=[('active', '#b68169')])
        self.demoStyle.configure('recordOnSmall.TButton', foreground='black', background='#b68169',font=self.font.buttonSmall)
        self.demoStyle.map("recordOnSmall.TButton",background=[('active', '#b68169')])
        self.demoStyle.configure('play.TButton', foreground='black', background='white', font=self.font.button)
        self.demoStyle.map("play.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('playSmall.TButton', foreground='black', background='white', font=self.font.buttonSmall)
        self.demoStyle.map("playSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('analysis.TButton', foreground='black', background='white', font=self.font.button)
        self.demoStyle.map("analysis.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('analysisSmall.TButton', foreground='black', background='white', font=self.font.buttonSmall)
        self.demoStyle.map("analysisSmall.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('exit.TButton', foreground='#7d150e', background='white', font=self.font.button)
        self.demoStyle.map("exit.TButton",background=[('active', '#7cb669')])
        self.demoStyle.configure('exitSmall.TButton', foreground='#7d150e' , background='white', font=self.font.buttonSmall)
        self.demoStyle.map("exitSmall.TButton",background=[('active', '#7cb669')])      
                
        # Progress bar
        self.demoStyle.configure("entryConf.Vertical.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        self.demoStyle.configure("entryConfSmall.Vertical.TProgressbar", foreground='black', background='#7d89af',
                            troughcolor='white',relief='sunken')
        
        
 
    class ourFonts():
        def __init__(self):             
            self.title = font.Font(family='Helvetica', size=24, weight='bold')
            self.titleSmall = font.Font(family='Helvetica', size=12, weight='bold')
            self.selectSound = font.Font(family='Helvetica', size=14)
            self.selectSoundSmall = font.Font(family='Helvetica', size=8)
            self.selection = font.Font(family='Courier', size=14)
            self.selectionSmall = font.Font(family='Courier', size=8)
            self.button = font.Font(family='Helvetica', size=14)
            self.buttonSmall = font.Font(family='Helvetica', size=8)
            self.labelEntry = font.Font(family='Helvetica', size=16)
            self.labelEntrySmall = font.Font(family='Helvetica', size=9)
            self.labelEntryRank = font.Font(family='Courier', size=18,weight='bold')
            self.labelEntryRankSmall = font.Font(family='Courier', size=10,weight='bold')
            

       
    def apply_styles(self,overall_size):
        
        if overall_size == 'normal':
            self.bannerFrame['style'] = 'banner.TFrame'
            self.bannerLabel['style'] = 'banner.TLabel'
           
            self.titleFrame['style'] = 'title.TFrame'
            self.titleLabel['style'] = 'title.TLabel'
            
            self.acquisitionFrame['style'] = 'acquisition.TFrame'
            self.loadButton['style'] = 'load.TButton'
            if not(self.recording):
                self.recordButton['style'] = 'recordOff.TButton'
            else:
                self.recordButton['style'] = 'recordOn.TButton'
            self.playButton['style'] = 'play.TButton'
            
            self.selectionFrame['style'] = 'selection.TFrame'
            self.selectSoundLabel['style'] = 'selectSound.TLabel'
            self.selectionLabel['style'] = 'selection.TLabel'
            
            self.analysisFrame['style'] = 'analysis.TFrame' 
            self.analysisButton['style'] = 'analysis.TButton'             
                           
            self.resultsOverallFrame['style'] = 'resultsOverall.TFrame'
            for n in range(self.numberOfClasses):
                self.resultsColumnFrame[n]['style'] = 'resultsColumn.TFrame'
                self.resLabelEntryLabel[n]['style'] = 'entryResults.TLabel'
                if self.numberOneInRanking != 0:
                    self.resLabelEntryLabel[self.numberOneInRanking].configure(style='entryResultsBest.TLabel')
                self.resRankEntryLabel[n]['style'] = 'entryResultsRank.TLabel'
                self.resConfEntryLabel[n]['style'] = "entryConf.Vertical.TProgressbar"
                                      
            self.exitFrame['style'] = 'exit.TFrame'
            self.exitButton['style'] = 'exit.TButton'
            
            self.logosFrame['style'] = 'logos.TFrame'
            for l in range(len(self.logo)):  
                self.logoLabel[l]['style'] = 'logos.TLabel'
        
        elif overall_size == 'small':
            self.bannerFrame['style'] = 'banner.TFrame'
            self.bannerLabel['style'] = 'banner.TLabel'
           
            self.titleFrame['style'] = 'title.TFrame'
            self.titleLabel['style'] = 'titleSmall.TLabel'
            
            self.acquisitionFrame['style'] = 'acquisition.TFrame'
            self.loadButton['style'] = 'loadSmall.TButton'
            if not(self.recording):
                self.recordButton['style'] = 'recordOffSmall.TButton'
            else:
                self.recordButton['style'] = 'recordOnSmall.TButton'
            self.playButton['style'] = 'playSmall.TButton'
            
            self.selectionFrame['style'] = 'selection.TFrame'
            self.selectSoundLabel['style'] = 'selectSoundSmall.TLabel'
            self.selectionLabel['style'] = 'selectionSmall.TLabel'
            
            self.analysisFrame['style'] = 'analysis.TFrame' 
            self.analysisButton['style'] = 'analysisSmall.TButton'             
                           
            self.resultsOverallFrame['style'] = 'resultsOverall.TFrame'
            for n in range(self.numberOfClasses):
                self.resultsColumnFrame[n]['style'] = 'resultsColumn.TFrame'
                self.resLabelEntryLabel[n]['style'] = 'entryResultsSmall.TLabel'
                if self.numberOneInRanking != 0:
                    self.resLabelEntryLabel[self.numberOneInRanking].configure(style='entryResultsBestSmall.TLabel')
                self.resRankEntryLabel[n]['style'] = 'entryResultsRankSmall.TLabel'
                self.resConfEntryLabel[n]['style'] = "entryConfSmall.Vertical.TProgressbar"
                                      
            self.exitFrame['style'] = 'exit.TFrame'
            self.exitButton['style'] = 'exitSmall.TButton'
            
            self.logosFrame['style'] = 'logos.TFrame'
            for l in range(len(self.logo)):  
                self.logoLabel[l]['style'] = 'logos.TLabel'        
            
        else:
            raise Exception('Unkown overall size {0}!'.format(overall_size))    
            

    def apply_packing(self,overall_size):
        
        if overall_size == 'normal':
            self.bannerFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0) 
            self.bannerLabel.pack(side='left', expand=True, padx=0, pady=0)
    
            self.titleFrame.pack(side='top', fill="both", expand=True, padx=10, pady=10) 
            self.titleLabel.pack(side='left', expand=True, padx=20, pady=20)
    
            self.acquisitionFrame.pack(side='top', fill="both", expand=True, padx=10, pady=0)        
            self.loadButton.pack(side='left', fill="both", expand=True, padx=20, pady=5, ipadx=10, ipady=10)
            self.recordButton.pack(side='left', fill="both", expand=True, padx=20, pady=5, ipadx=10, ipady=10)
            self.playButton.pack(side='left', fill="both", expand=True, padx=20, pady=5, ipadx=10, ipady=10)
            
            self.selectionFrame.pack(side='top', fill="y", expand=True, padx=10, pady=10) 
            self.selectSoundLabel.pack(side='left', expand=True, padx=10, pady=10)
            self.selectionLabel.pack(side='left', expand=True, padx=10, pady=10)
            
            self.analysisFrame.pack(side='top', expand=True, padx=10, pady=0)      
            self.analysisButton.pack(padx=20, expand=True, pady=10, ipadx=80, ipady=10)
            
            self.resultsOverallFrame.pack(expand=True, padx=20, pady=10) 
            for n in range(self.numberOfClasses):
                self.resLabelEntryLabel[n].pack(side='top', fill="both", expand=True, padx=5, pady=10)
                self.resRankEntryLabel[n].pack(side='top', fill="y", expand=True, padx=5, pady=10)
                self.resConfEntryLabel[n].pack(side='bottom', fill="y", expand=True, padx=5, pady=10)
                self.resultsColumnFrame[n].pack(side='left', fill="both", expand=True, padx=10, pady=10)
            
            self.exitFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0)
            self.exitButton.pack(side='left', expand=True, padx=120, pady=20, ipadx=80, ipady=10)
            
            self.logosFrame.pack(side='bottom', expand=True, fill="both", padx=10, pady=20)
            for l in range(len(self.logo)):  
                self.logoLabel[l].pack(side='left', expand=True, padx=10, pady=0)
                
        elif overall_size == 'small':
            self.bannerFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0) 
            self.bannerLabel.pack(side='left', expand=True, padx=0, pady=0)
    
            self.titleFrame.pack(side='top', fill="both", expand=True, padx=5, pady=5) 
            self.titleLabel.pack(side='left', expand=True, padx=10, pady=10)
    
            self.acquisitionFrame.pack(side='top', fill="both", expand=True, padx=5, pady=0)        
            self.loadButton.pack(side='left', fill="both", expand=True, padx=10, pady=2.5, ipadx=5, ipady=5)
            self.recordButton.pack(side='left', fill="both", expand=True, padx=10, pady=2.5, ipadx=5, ipady=5)
            self.playButton.pack(side='left', fill="both", expand=True, padx=10, pady=2.5, ipadx=5, ipady=5)
            
            self.selectionFrame.pack(side='top', fill="y", expand=True, padx=5, pady=5) 
            self.selectSoundLabel.pack(side='left', expand=True, padx=5, pady=5)
            self.selectionLabel.pack(side='left', expand=True, padx=5, pady=5)
            
            self.analysisFrame.pack(side='top', expand=True, padx=5, pady=0)      
            self.analysisButton.pack(padx=20, expand=True, pady=5, ipadx=40, ipady=5)
            
            self.resultsOverallFrame.pack(expand=True, padx=10, pady=5) 
            for n in range(self.numberOfClasses):
                self.resLabelEntryLabel[n].pack(side='top', fill="both", expand=True, padx=2.5, pady=5)
                self.resRankEntryLabel[n].pack(side='top', fill="y", expand=True, padx=2.5, pady=5)
                self.resConfEntryLabel[n].pack(side='bottom', fill="y", expand=True, padx=2.5, pady=5)
                self.resultsColumnFrame[n].pack(side='left', fill="both", expand=True, padx=5, pady=5)
            
            self.exitFrame.pack(side='top', fill="both", expand=True, padx=0, pady=0)
            self.exitButton.pack(side='left', expand=True, padx=60, pady=10, ipadx=40, ipady=5)
            
            self.logosFrame.pack(side='bottom', expand=True, fill="both", padx=5, pady=10)
            for l in range(len(self.logo)):  
                self.logoLabel[l].pack(side='left', expand=True, padx=5, pady=0)
            
        else:
            raise Exception('Unkown overall size {0}!'.format(overall_size))

            
    def image_selection(self,overall_size):  
        
        if overall_size == 'normal':
            self.bannerLabel['image'] = self.banner
            for l in range(len(self.logo)):  
                self.logoLabel[l]['image'] = self.logo[l]              
        elif overall_size == 'small':
            self.bannerLabel['image'] = self.bannerSmall
            for l in range(len(self.logo)):  
                self.logoLabel[l]['image'] = self.logoSmall[l]                     
        else:
            raise Exception('Unkown overall size {0}!'.format(overall_size))    


    def load_sound(self):
        if not(self.recording):
            
            self.soundFilename = filedialog.askopenfilename(initialdir = self.defaultSoundDataPath,
                                    title = "Select file", filetypes = (("WAV files","*.wav"),("All files","*.*")))
            
            if not(self.soundFilename == ''):  # if not canceled by user
                self.audioAnalysis.load(self.soundFilename)
                
                pathStr, fileStr = path.split(self.soundFilename)
                self.soundName.set(fileStr)
                self.resetResultDisplay()

        else:
            print('Busy')
            
           
        
    def record_sound(self):
        if not(self.recording):
            
            if self.overallSize == 'normal':
                self.recordButton.configure(style='recordOn.TButton',text='Recording')
            elif self.overallSize == 'small':
                self.recordButton.configure(style='recordOnSmall.TButton',text='Recording')
            else:
                raise Exception('Unkown overall size {0}!'.format(self.overallSize))
            self.recordButton.update()
            
            self.recording = True 
            startTime = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
            
            self.audioAnalysis.record()
                 
            self.recording = False
            if self.overallSize == 'normal':
                self.recordButton.configure(style='recordOff.TButton',text='Record')
            elif self.overallSize == 'small':
                self.recordButton.configure(style='recordOffSmall.TButton',text='Record')
           
            
            self.soundFilename = None
            self.resetResultDisplay()
            self.soundName.set('Rec - ' + startTime)
        
        
    def play_sound(self):
        if not(self.recording):
            self.audioAnalysis.play()
     
        
        
    def track_overall_size(self,event):  
        
         change = False
         
         if self.overallSize == 'small':
             if event.width > self.masterFrameSwitchWidth and event.height > self.masterFrameSwitchHeight:
                 change = True 
         elif self.overallSize == 'normal':     
             if event.width < self.masterFrameSwitchWidth or event.height < self.masterFrameSwitchHeight:   
                 change = True 
         else:
              raise Exception('Unkown overall size {0}!'.format(self.overallSize))       
             
         if change:
             self.switch_overall_size() 
            
            
         
    def switch_overall_size(self):  
       
        if self.overallSize == 'small':
            NEW_SIZE = 'normal'
            self.overallSize = 'normal'
        elif self.overallSize == 'normal':
            NEW_SIZE = 'small'
            self.overallSize = 'small'
        else:
            raise Exception('Unkown overall size {0}!'.format(self.overallSize))   
            
        self.image_selection(NEW_SIZE)
        self.apply_styles(NEW_SIZE)
        self.apply_packing(NEW_SIZE)     
            
            
        
    def run_analysis(self):
        
        if not(self.recording):
            
            if not(self.audioAnalysis.data is None):
            
                prob, predict_idxs = self.audioAnalysis.analysis()
     
                self.resultRank[0].set(np.array2string(np.where(predict_idxs==0)[0][0]+1))
                self.resultRank[1].set(np.array2string(np.where(predict_idxs==1)[0][0]+1))
                self.resultRank[2].set(np.array2string(np.where(predict_idxs==2)[0][0]+1))
                self.resultRank[3].set(np.array2string(np.where(predict_idxs==3)[0][0]+1))
                self.resultRank[4].set(np.array2string(np.where(predict_idxs==4)[0][0]+1))
                
                self.posthocProb[0].set(prob[0])
                self.posthocProb[1].set(prob[1])
                self.posthocProb[2].set(prob[2])
                self.posthocProb[3].set(prob[3])
                self.posthocProb[4].set(prob[4])
                
                self.numberOneInRanking = np.argmax(prob);
                
                if self.overallSize == 'normal':                                 
                    self.resLabelEntryLabel[self.numberOneInRanking].configure(style='entryResultsBest.TLabel')
                elif self.overallSize == 'small':
                    self.resLabelEntryLabel[self.numberOneInRanking].configure(style='entryResultsBestSmall.TLabel')
                else:
                    raise Exception('Unkown overall size {0}!'.format(self.overallSize))    
     
        
    def resetResultDisplay(self): 
        
        for k in range(len(self.resLabelEntryLabel)):
            self.numberOneInRanking = 0;
            if self.overallSize == 'normal':
                self.resLabelEntryLabel[k].configure(style='entryResults.TLabel')
            elif self.overallSize == 'small':   
                self.resLabelEntryLabel[k].configure(style='entryResultsSmall.TLabel')
            else:
                raise Exception('Unkown overall size {0}!'.format(self.overallSize))   
            self.resultRank[k].set(0)
            self.posthocProb[k].set(0)


        
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
    # demo.master.geometry("600x830")
    demo.mainloop()
    
   
if __name__ == '__main__':
    
    main() 
