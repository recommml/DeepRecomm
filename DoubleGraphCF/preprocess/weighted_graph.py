''' 
used to generate weighted graphs for users and items respectively
'''
import numpy as np
import scipy.sparse as sp

dataset = 'amazon-book'
data_path = 'Data'

def safe_append(dic, key, val):
    if key not in dic:
        dic[key] = []
    dic[key].append(val)

def union_count(arr1, arr2):
    cnt = 0
    for val1 in arr1:
        if val1 in arr2:
            cnt += 1
    return cnt

def generate_weighted_graph(dataset='amazon-book', data_path='Data'):
    # get num_users, num_items
    root_path = data_path + '/' + dataset + '/'
    fuser = open(root_path + 'user_list.txt')
    fitem = open(root_path + 'item_list.txt')
    num_users, num_items = -1, -1
    luser, litem = fuser.readline().replace('\n', ''), fitem.readline().replace('\n', '')
    while luser is not None and luser != '':
        num_users += 1
        luser = fuser.readline().replace('\n', '')
    while litem is not None and litem != '':
        num_items += 1
        litem = fitem.readline().replace('\n', '')
    fuser.close()
    fitem.close()

    # get adjacent matrix
    users_map = {}
    items_map = {}
    ftrain = open(root_path + 'train.txt')
    line = ftrain.readline().replace('\n', '')
    while line is not None and line != '':
        cols = line.split(' ')
        len_item = len(cols) - 1
        cur_user = int(cols[0])
        for i in range(len_item):
            if cols[i + 1] == '':
                continue
            cur_item = int(cols[i + 1])
            safe_append(users_map, cur_user, cur_item)
            safe_append(items_map, cur_item, cur_user)
        line = ftrain.readline().replace('\n', '')
    ftrain.close()
    ftest = open(root_path + 'test.txt')
    line = ftest.readline().replace('\n', '')
    while line is not None and line != '':
        cols = line.split(' ')
        len_item = len(cols) - 1
        cur_user = int(cols[0])
        for i in range(len_item):
            if cols[i + 1] == '':
                continue
            cur_item = int(cols[i + 1])
            safe_append(users_map, cur_user, cur_item)
            safe_append(items_map, cur_item, cur_user)
        line = ftest.readline().replace('\n', '')
    ftest.close()

    # generate isomorphic graphs on users and items respectively
    user_graph = sp.dok_matrix((num_users, num_users), dtype=np.float32)
    item_graph = sp.dok_matrix((num_items, num_items), dtype=np.float32)
    user_graph = user_graph.tolil()
    item_graph = item_graph.tolil()
    for cur_user in range(num_users):
        if cur_user % 100 == 0:
            print('for user', cur_user)
        cur_items = users_map[cur_user]
        for cur_item in cur_items:
            target_users = items_map[cur_item]
            for target_user in target_users:
                user_graph[cur_user, target_user] += 1
    for cur_item in range(num_items):
        if cur_item % 100 == 0:
            print('for item', cur_item)
        cur_users = items_map[cur_item]
        for cur_user in cur_users:
            target_items = users_map[cur_user]
            for target_item in target_items:
                item_graph[cur_item, target_item] += 1
    print("user graph:", user_graph.count_nonzero(), "; item graph:", item_graph.count_nonzero())
    user_graph = user_graph.tocsr()
    item_graph = item_graph.tocsr()
    sp.save_npz(root_path + 'user_graph.npz', user_graph)
    sp.save_npz(root_path + 'item_graph.npz', item_graph)

if __name__ == '__main__':
    generate_weighted_graph()