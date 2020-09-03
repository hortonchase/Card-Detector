# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:27:39 2020

@author: Chase
"""
import cv2
import imutils
import getPerspective
from skimage.measure import compare_ssim
import sys
import glob
import os
#code to load in all the comparison images
ranks = []
suits = []
rankFiles = [
    "pictures/rank/a.jpg", "pictures/rank/k.jpg", "pictures/rank/q.jpg", "pictures/rank/j.jpg", "pictures/rank/ten.jpg",
    "pictures/rank/nine.jpg", "pictures/rank/eight.jpg", "pictures/rank/seven.jpg", "pictures/rank/six.jpg", "pictures/rank/five.jpg",
    "pictures/rank/four.jpg", "pictures/rank/three.jpg", "pictures/rank/two.jpg"]
suitFiles = ["pictures/suit/spades.jpg", "pictures/suit/clubs.jpg", "pictures/suit/diamonds.jpg", "pictures/suit/hearts.jpg"]
#def getCard(rank, suit):
for myFile in rankFiles:
    image = cv2.imread(myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ranks.append(image)
for myFile in suitFiles:
    image = cv2.imread(myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    suits.append(image)



#function to easily display images
def display(image):
    cv2.imshow("Image", imutils.resize(image.copy(), height = 720))
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        vs.release()
        sys.exit()
    cv2.destroyAllWindows
#image = cv2.imread('pictures/cards/card7.jpg')
vs = cv2.VideoCapture(0)
while True:
    image = vs.read()[1]
    original = image.copy()
    #resize window
    ratio = image.shape[0]/720
    image = imutils.resize(image,height = 720)
    
    display(image)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    display(gray)
    
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
 #   for x in range(180, 254):
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    
    display(thresh)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    if(len(cnts) > 0):
        cardContour = max(cnts, key = cv2.contourArea)
        peri = cv2.arcLength(cardContour, True)
        approx = cv2.approxPolyDP(cardContour, 0.02 * peri, True)
        if len(approx) == 4:
        
            cv2.drawContours(image, [cardContour], -1, (0, 255, 0), 2)
            
            display(image)
            
            points = image.copy()
            approx = approx.reshape(4,2)
            cv2.circle(points, (int(approx[0, 0]), int(approx[0, 1])), 8, (0, 0, 255), -1) #red TL
            cv2.circle(points, (int(approx[1, 0]), int(approx[1, 1])), 8, (0, 255, 0), -1) #green TR
            cv2.circle(points, (int(approx[2, 0]), int(approx[2, 1])), 8, (255, 0, 0), -1) #Blue BR
            cv2.circle(points, (int(approx[3, 0]), int(approx[3, 1])), 8, (255, 255, 0), -1)#Cyan BL
            
            display(points)

            flat = getPerspective.four_point_transform(original.copy(), approx*ratio)
            display(flat)
            if(flat.shape[0] < flat.shape[1]):
                flat = imutils.rotate_bound(flat, 90)
            display(flat)
            
            gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
            
            display(gray)
            
            flatThresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
            
            display(flatThresh)
            
            cropped = flatThresh[:int(flatThresh.shape[0]/2.5), :int(flatThresh.shape[1]/3.7)]
            
            display(cropped)
            
            #cv2.imshow("flat", imutils.resize(flat, height = 720))
            #cv2.imshow("gray", imutils.resize(gray, height = 720))
            #cv2.imshow("thresh", imutils.resize(flatThresh, height = 720))
            #cv2.imshow("cropped", imutils.resize(cropped, height = 720))
            rank = cropped[:int(cropped.shape[0]/1.85)]
            
            suit = cropped[int(cropped.shape[0]/1.85):]
            
            
            cv2.imshow("suit", imutils.resize(suit, height = 720))
            cv2.imshow("rank", imutils.resize(rank, height = 720))
            cv2.waitKey(0)
            
            
            #cv2.imshow("suit", imutils.resize(suit, height = 720))
            #cv2.imshow("rank", imutils.resize(rank, height = 720))
            
            
            #finding contours in suit and rank then drawing a box around them and cropping
            cnts = cv2.findContours(suit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key = cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)
            
            
            
            suitSquare = cv2.rectangle(suit.copy(), (x , y ), (x + w, y + h), (255, 0, 127))
            
            
            
            suit = suit[y:y+h, x:x+w]
            
            
            cnts = cv2.findContours(rank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key = cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            rankSquare = cv2.rectangle(rank.copy(), (x , y ), (x + w, y + h), (95, 191, 0))
            cv2.imshow("suit", imutils.resize(suitSquare, height = 720))
            cv2.imshow("rank", imutils.resize(rankSquare, height = 720))


            cv2.waitKey(0)
            cv2.destroyAllWindows()
            

            rank = rank[y:y+h, x:x+w]
            
            
            suit = imutils.resize(suit, height = 850)
            rank = imutils.resize(rank, height = 850)
            
            
            
            cv2.imshow("suit", suit)
            cv2.imshow("rank", rank)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
            rankNames = ["Ace", "King", "Queen", "Jack", "Ten", "Nine", "Eight", "Seven", "Six", "Five", "Four", "Three", "Two"]
            suitNames = ["Spades", "Clubs", "Diamonds", "Hearts"]
            i = 0
            bestGuess = [-5, "", -5, ""]
    
            for r in ranks:
                test = cv2.resize(r, (rank.shape[1], rank.shape[0]))
                score, diff = compare_ssim(test, rank, full = True) 
                diff = (diff * 255).astype("uint8")
                # print("SSIM of {}: {}".format(suitNames[i], score))
                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                display(thresh)
                if score > bestGuess[0]:
                    bestGuess[0] = score
                    bestGuess[1] = rankNames[i]
                i+=1
            i = 0
            for s in suits:
                test = cv2.resize(s, (suit.shape[1], suit.shape[0]))
                score, diff = compare_ssim(test, suit, full = True)
                diff = (diff * 255).astype("uint8")
                # print("SSIM of {}: {}".format(suitNames[i], score))
                thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                display(thresh)
                if score > bestGuess[2]:
                    bestGuess[2] = score
                    bestGuess[3] = suitNames[i]
                i+=1
      #  print("The card is the {} of {}.".format(bestGuess[1], bestGuess[3]))
            M = cv2.moments(cardContour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            final = cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
            cv2.putText(final, ("{} of {}".format(bestGuess[1], bestGuess[3])), (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(final, "Certainty: {:.2%}".format(bestGuess[0] * bestGuess[2]), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.destroyAllWindows()
            cv2.imshow("final", final)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
vs.release()    
cv2.destroyAllWindows()