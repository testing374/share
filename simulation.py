## 賽道長24格, X 位置 Y 高度
## 例如 [[1,2,3,4,5,6],...] 6最高 1最低

from random import randint, random

board = [[]]*30


## 帶住上面的人移動
def carry(self, forward):
    ## 自己本來的高度 0最低
    y = board[self.x].index(self)
    ## 目的地本來的人數
    tmp = len(board[self.x + forward])
    ## 目的地增加
    board[self.x + forward] += board[self.x][y:]
    ## 本來位置刪除
    board[self.x] = board[self.x][:y]
    ## 更新移動的人的x
    for i in board[self.x + forward][tmp:]:
        i.x += forward



class Person:
    def __init__(self):
        self.x = 0
        self.wins = 0


class Jinhsi(Person):
    def move(self):
        if random() < 0.4:
            ## 自己本來的高度
            y = board[self.x].index(self)
            board[self.x] += [self]
            del board[self.x][y]
        carry(self, randint(1,3))


class Calcharo(Person):
    def move(self):
        dice = randint(1,3)
        last = True
        for i in p:
            if i.x < self.x:
                last = False
                break
        forward = dice + 3 if last else dice
        carry(self, forward)


class Shorekeeper(Person):
    def move(self):
        carry(self, randint(2,3))




class Camellya(Person):
    def move(self):
        dice = randint(1,3)
        if random() < 0.5:
            dice += len(board[self.x])-1
            ## 更新自己位置
            y = board[self.x].index(self)
            board[self.x + dice] += [self]
            del board[self.x][y]
            self.x += dice
            
        else: carry(self, dice)



class Carlotta(Person):
    def move(self):
        dice = randint(1,3)
        forward = dice*2 if random() < 0.28 else dice
        carry(self, forward)


## 6個參加者
p = [None]*5
p[0] = Jinhsi()
p[1] = Calcharo()
p[2] = Shorekeeper()
p[3] = Camellya()
p[4] = Carlotta()


board[0] = [p[i] for i in (4,3,2,1,0)]   ## p[0] on top
end = False

## Start a game
while not end:
    for i in range(5):
        print(i)
        p[i].move()
        if p[i].x >= 24:
            end = True
            p[i].wins += 1
            break

print([p[i].wins for i in range(5)])




## 今汐 頭頂有其他, 40%移到頂
## 長離 下有其他, 下回合65%最後行
## 卡卡羅 如最後一名, 額外前進三格
## 守岸 只動2/3
## 椿 50% 每多一人, 額外+1 而且不帶其他人
## 珂萊塔 28% 步數*2

'''
布蘭特  如第一個行，額外＋２
坎特蕾拉  一場只一次：移動中首次遇別人，堆疊一起行
菲比  50%額外前進一格
卡提希婭  一場只一次：自身移動後，如最後一名，剩餘回合６０％額外前進２格
洛可可  如最後一個行, 額外前進2格
贊尼  只擲1/3。開始移動時, 如堆疊, 下回合40%額外前進2格
'''

## Jinhsi Changli Calcharo Shorekeeper Camellya Carlotta
##

