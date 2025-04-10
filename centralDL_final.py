## 地址可能是車位

from selenium import webdriver

from time import sleep

from selenium.webdriver import Keys, ActionChains
from selenium.webdriver.common.by import By

import pyperclip
import tkinter
from random import randint, random

def foo(page):
    ss = ''
    page = int(page)
    ele = driver.find_element(By.CSS_SELECTOR, "body")
    ele.send_keys(Keys.CONTROL,'a')
    sleep(1)

    for p in range(page):
        ele.send_keys(Keys.CONTROL,'c')

        d = pyperclip.paste()
        fn = d.split(' 成交紀錄')[0].split('最新成交')[1][2:]
        d = d.split('資料來源')[1].split('精選居屋')[0].split('\n')
        special = True if '特殊成交個案' in d else False
        del d[0]
        del d[-7 if special else -6:]
        d = map(lambda x: x[:-1], d)
        d = list(filter(lambda x: x!='\ue626', d))
        s = ''
        for i in range(len(d)//7):
            j = i*7
            s += '{},{},{},{},{},{}\n'.format(d[j],d[j+1],d[j+2][1:].replace(',',''),d[j+3],d[j+4][:-1].replace(',',''),d[j+5][2:].replace(',',''))
## d[j+2][1:-1] 買   d[j+2][1:] 租
        ss += s
        if p != page-1:
            driver.find_element(By.CSS_SELECTOR, "button.btn-next").click()
        sleep(2+random())#randint(4,6))


    with open(fn+'.txt', 'w', encoding='utf-8') as f:
        #f.write('日期,地址,成交價,升跌,面積(實),呎價(實)\n')
        f.write('Date,Address,Price,Changes,Saleable Area,Unit Rate\n')
        f.write(ss)
    print(fn+' 完成')



driver = webdriver.Firefox()



tk = tkinter.Tk()
tk.wm_attributes("-topmost", 1)
tk.geometry("250x100")
txt = tkinter.Entry(tk, width=20)
txt.pack(padx=10, pady=10, side="top")
btn = tkinter.Button(tk, text ="複制", command = lambda: foo(txt.get()), width=200, height=100)
btn.pack(padx=10, pady=10, side="bottom")
tk.mainloop()



#url = "https://hk.centanet.com/findproperty/list/transaction?q=a1eb51674df"
#url = 'https://hk.centanet.com/findproperty/list/transaction?q=a19545deebe'



