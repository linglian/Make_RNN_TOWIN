#coding=utf-8

import requests
import bs4
import datetime

def getLongHuDou():
    
    url = 'http://potal.377kx.com/?game_id=402&page={}'
    
    end_result_list = []
    
    def get_longhudou_of_page(page):
        print '访问页面...'
        response = requests.get(url.format(page))
        print '访问成功...'
        soup = bs4.BeautifulSoup(response.text)
        span = soup.select('table tr td span')
        def getResult(name):
            if name == 'long':
                return 0
            elif name == 'hu':
                return 1
            elif name == 'he':
                return 2
            else:
                raise ValueError('什么乱七八糟的参数都往里传啊，老铁', name)
        result_list = []
        for i in range(0, 12):
            for j in range(0, 6):
                try:
                    result_list.append([getResult(span[j * 12 + i].get('class')[1])])
                except Exception, ex:
                    continue
        if len(result_list) == 0:
            raise ValueError('行了，没了，别读了')
        return result_list
    
    try:
        for i in range(1, 100):
            end_result_list.extend(get_longhudou_of_page(i))
            print '已经获取%d' % len(end_result_list)
    except Exception, ex:
        print 'end page {} because error {}'.format(i, ex.message)
    return end_result_list
end_result_list = getLongHuDou()
print len(end_result_list)
print end_result_list
import numpy as np
if len(end_result_list) != 0:
    np.save('./龙虎斗平民场_{}.npy'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')), end_result_list)