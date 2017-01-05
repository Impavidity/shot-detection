#!/usr/bin/python

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

__bin__ = 64
__slot__ = 1000
__color__ = [(255,0,0),(0,255,0),(0,0,255)]
__absolute_threshold__ = 1.0

class shotDetector(object):
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.myAns = []
        self.gold = []
        self.threshold = 1.0
        self.thList = []
        self.fList = []
        fans = open("testAns")
        for line in fans.readlines():
            self.gold.append(int(line.strip()))

    def visualize(self, chists):
        h = np.zeros((256,__bin__,3))
        bin = np.arange(__bin__).reshape(__bin__,1)
        i = 0
        for item in chists:
            original = item * 256
            hist = np.int32(np.around(original))
            pts = np.column_stack((bin, hist))
            cv2.polylines(h, [pts], False, __color__[i])
            i += 1
        h = np.flipud(h)
        cv2.imshow('color',h)
        cv2.waitKey(0)

    def visThread(self, score):
        plt.plot(range(len(score)), score, "b*")
        plt.title("Similarity")
        plt.legend()
        plt.show()

    def visResult(self, threadList, FScore):
        print "Max F score:",max(FScore)
        print "threshold:", threadList[FScore.index(max(FScore))]
        plt.plot(threadList, FScore, "b*")
        plt.title("threshold-F")
        plt.legend()
        plt.show()

        

    def run(self):
        assert(self.video_path is not None), "You should must the video paht"

        cap = cv2.VideoCapture(self.video_path)
        hists = []
        frames = []
        count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            chists = [cv2.calcHist([frame], [c], None, [64], [0,256]) for c in range(3)]
            for item in chists:
                cv2.normalize(item, item, 0, 1, cv2.NORM_MINMAX)
            chists = np.array([chist for chist in chists])
            #print chists
            hists.append(chists.flatten())
            #self.visualize(chists)
            count += 1
            if count%100 == 0:
                print count
            
        #print hists

        cap.release()

        #Compute the score
        scores = [cv2.compareHist(pair[0], pair[1], cv2.cv.CV_COMP_INTERSECT)/sum(pair[0]) for pair in zip(hists[:-1],hists[1:])]
        #print scores
        #self.visThread(scores)
        #print "most similar: ", max(scores), " least similar: ", min(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        self.thList = []
        self.fList = []

        step = float(3)/float(__slot__)
        for i in range(__slot__):
            threshold = mean_score - i * step * std_score
            self.thList.append(threshold)
            self.myAns = []
            for i in range(len(scores)):
                if scores[i] < threshold:
                    self.myAns.append(i)
            self.test()
        self.visResult(self.thList, self.fList)


    def test(self):
        correct = 0
        for i in range(len(self.myAns)):
            if (self.gold.count(self.myAns[i]+2)>0):
                correct += 1
        
        P = float(correct)/float(len(self.myAns))
        R = float(correct)/float(len(self.gold))
        F = 2*P*R/(P+R)
        #print "P: ",P, " R: ",R, " F: ",F
        #self.fout.write(str(P)+" "+str(R)+" "+str(F)+"\n")
        self.fList.append(F)


    def dist(self, vec1, vec2):
        ans = 0
        if len(vec1)==len(vec2):
            for i in range(len(vec1)):
                ans += (vec1[i]-vec2[i])**2
        return ans


    def HU(self):
        cap = cv2.VideoCapture(self.video_path)
        vector = []
        count = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            item = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            moments = cv2.moments(item)
            hu_moments = cv2.HuMoments(moments)
            vector.append([hu_moments[0][0],hu_moments[1][0],hu_moments[2][0]])
            count += 1
            if (count%100==0):
                print count

        scores = []
        for i in range(len(vector)-1):
            sim = self.dist(vector[i],vector[i+1])
            scores.append(sim)
        #self.visThread(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        step = (mean_score/std_score)/float(__slot__)
        self.thList = []
        self.fList = []
        for i in range(__slot__):
            threshold = mean_score - i * step * std_score
            #print "Thread: ",threshold
            self.thList.append(threshold)
            self.myAns = []
            for i in range(len(scores)):
                if scores[i] > threshold:
                    self.myAns.append(i)
            self.test()
        self.visResult(self.thList, self.fList)

        





        



if __name__=="__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "usage: python ./shotDetect.py <video-path>"
        sys.exit()
    video_path = sys.argv[1]
    detector = shotDetector(video_path=video_path)
    #detector.run()
    detector.HU()
    #detector.test()