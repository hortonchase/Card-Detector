import cv2
import imutils
import getPerspective
from skimage.measure import compare_ssim
import sys

#code to load in all the comparison images
ranks = []
cardsDrawn = []
lst = []
TOTALCOUNT = 0
TOTAL = 0
suits = []
card_cnts = []
bestGuess = None
rankNames = ["Ace", "King", "Queen", "Jack", "Ten", "Nine", "Eight", "Seven", "Six", "Five", "Four", "Three", "Two"]
suitNames = ["Spades", "Clubs", "Diamonds", "Hearts"]
addValues = [11, 10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2]
countValues = [-1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1, 1, 1]

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
    cv2.waitKey(0)
    cv2.destroyAllWindows

#image = cv2.imread('pictures/cards/card7.jpg')
vs = cv2.VideoCapture(0)
while True:
    image = vs.read()[1]
    original = image.copy()

    #resize window
    ratio = image.shape[0]/720
    image = imutils.resize(image,height = 720)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    final = imutils.resize(image.copy(), height = 720)
    
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
 #   for x in range(180, 254):
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:7]
    if(len(cnts) > 0):
        for cardContour in cnts:
            peri = cv2.arcLength(cardContour, True)
            approx = cv2.approxPolyDP(cardContour, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(cardContour) >= 90000:
                cv2.drawContours(image, [cardContour], -1, (0, 255, 0), 2)
                flat = getPerspective.four_point_transform(original.copy(), approx.reshape(4,2)*ratio)
                if(flat.shape[0] < flat.shape[1]):
                    flat = imutils.rotate_bound(flat, 90)
                gray = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
                flatThresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    
                cropped = flatThresh[:int(flatThresh.shape[0]/2.5), :int(flatThresh.shape[1]/3.7)]
                #cv2.imshow("flat", imutils.resize(flat, height = 720))
                #cv2.imshow("gray", imutils.resize(gray, height = 720))
                #cv2.imshow("thresh", imutils.resize(flatThresh, height = 720))
                #cv2.imshow("cropped", imutils.resize(cropped, height = 720))
                rank = cropped[:int(cropped.shape[0]/1.85)]
                suit = cropped[int(cropped.shape[0]/1.85):]
                #cv2.imshow("suit", imutils.resize(suit, height = 720))
                #cv2.imshow("rank", imutils.resize(rank, height = 720))
    
                #finding contours in suit and rank then drawing a box around them and cropping
                cnts = cv2.findContours(suit, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
                cnts = imutils.grab_contours(cnts)
                if len(cnts) == 0:
                    cv2.imshow("final", final)
                    break
                c = max(cnts, key = cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(c)
                suit = suit[y:y+h, x:x+w]
    
                cnts = cv2.findContours(rank, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
                cnts = imutils.grab_contours(cnts)
                if len(cnts) == 0:
                    cv2.imshow("final", final)
                    break
                c = max(cnts, key = cv2.contourArea)
                (x, y, w, h) = cv2.boundingRect(c)
                rank = rank[y:y+h, x:x+w]
    
    
                suit = imutils.resize(suit, height = 200)
                rank = imutils.resize(rank, height = 200)
                #cv2.imshow("suit", suit)
                #cv2.imshow("rank", rank)

                i = 0
                bestGuess = [-5, "", -5, "", 0]

                for r in ranks:
                    test = cv2.resize(r, (rank.shape[1], rank.shape[0]))
                    score, diff = compare_ssim(test, rank, full = True)
                    diff = (diff * 255).astype("uint8")
                   # print("SSIM of {}: {}".format(rankNames[i], score))
                    thresh = cv2.threshold(diff, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    if score > bestGuess[0]:
                        bestGuess[0] = score
                        bestGuess[1] = rankNames[i]
                        bestGuess[4] = i
                    i+=1
                i = 0
                for s in suits:
                    test = cv2.resize(s, (suit.shape[1], suit.shape[0]))
                    score, diff = compare_ssim(test, suit, full = True)
                    diff = (diff * 255).astype("uint8")
                   # print("SSIM of {}: {}".format(suitNames[i], score))
                    thresh = cv2.threshold(diff, 0, 255,
                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    if score > bestGuess[2]:
                        bestGuess[2] = score
                        bestGuess[3] = suitNames[i]
                    i+=1
                M = cv2.moments(cardContour)
                if M["m00"] != 0 :
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    final = cv2.circle(image, (cX, cY), 7, (0, 0, 255), -1)
                
                if bestGuess is not None:
                    cv2.putText(final, ("{} of {}".format(bestGuess[1], bestGuess[3])), (cX-20, cY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
 #                   cv2.putText(final, "Certainty: {:.2%}".format((bestGuess[0] + bestGuess[2])/2), (10, 20),
 #                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if (bestGuess[0] + bestGuess[2])/2 > .7 and (bestGuess[1] + bestGuess[3]) not in cardsDrawn:
                        temp = bestGuess[1] + bestGuess[3]
                        lst.append(temp)
                        if lst.count(temp) > 30:
                            for (temp) in lst:
                                lst.remove(temp)
                            cardsDrawn.append(bestGuess[1]+bestGuess[3])
                            TOTALCOUNT += countValues[bestGuess[4]]
                            TOTAL += addValues[bestGuess[4]]
                    cv2.putText(final, "Count: {}".format(TOTALCOUNT), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(cardsDrawn)
            cv2.imshow("final", final)
    else:
        cv2.imshow("final", final)
    k = cv2.waitKey(1)
    if k == 27:
        cv2.destroyAllWindows()
        vs.release()
        sys.exit()   
    if k == ord('q'):
        TOTAL = 0