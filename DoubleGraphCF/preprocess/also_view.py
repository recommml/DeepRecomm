import random

dataset = 'amazon-book'
data_path = 'Data/'
org_path = 'Data/origins/'

def proc_line(line, fdst):
    cols = line.split(' ')
    if line is None or line == '':
        return
    userid = cols[0]
    items = cols[1:]
    max_itemid = 0
    for itemid in items:
        if userid == '' or itemid == '':
            continue
        if int(itemid) > max_itemid:
            max_itemid = int(itemid)
        fdst.write(userid + '\t' + itemid + '\t1\t0\n')
    return max_itemid

def proc_negative(line, fneg, max_itemid):
    if line is None or line == '':
        return
    cols = line.split(' ')
    userid = cols[0]
    item_strs = cols[1:]
    items = []
    for istr in item_strs:
        if istr != '':
            items.append(int(istr))
    for itemid in items:
        fneg.write('(' + userid + ',' + str(itemid) + ')')
        cnt = 0
        while cnt < 99:
            ridx = random.randint(0, max_itemid)
            if ridx not in items:
                fneg.write('\t' + str(ridx))
                cnt += 1
        fneg.write('\n')

def proc(mode='train'):
    fsrc = open(org_path + dataset + '/' + mode + '.txt')
    fdst = open(data_path + dataset + '.' + mode + '.rating', 'w')
    max_itemid = 0
    sline = fsrc.readline().replace('\n', '')
    while sline is not None and sline != '':
        tmax = proc_line(sline, fdst)
        if max_itemid < tmax:
            max_itemid = tmax
        sline = fsrc.readline().replace('\n', '')
    fsrc.close()
    fdst.close()
    if mode == 'test':
        fneg = open(data_path + dataset + '.test.negative', 'w')
        fsrc = open(org_path + dataset + '/' + mode + '.txt')
        sline = fsrc.readline().replace('\n', '')
        while sline is not None and sline != '':
            proc_negative(sline, fneg, max_itemid)
            sline = fsrc.readline().replace('\n', '')

proc()
proc('test')