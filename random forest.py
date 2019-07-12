from sklearn.metrics import r2_score
import pandas as pd
import time as tm
import numpy as np
import multiprocessing
import math

from sklearn.model_selection import train_test_split

'''
决策树
'''
class decision_tree_regressor:

    def __init__(self, max_fec_num, max_depth):
        self.max_fec_num = max_fec_num
        self.max_depth = max_depth

    '''
    计算方差
    '''
    def cal_variance(self, label):
        return (np.var(label) * label.shape[0]).item()

    '''
    计算平均数
    '''
    def cal_mean(self, label):
        return np.mean(label)

    '''
    划分数据集
    '''
    def split_dataset(self, data, label, index, value):
        l = np.nonzero(data.iloc[:, index] < value)[0]
        r = np.nonzero(data.iloc[:, index] > value)[0]
        return data.iloc[l, :], label.iloc[l, :], data.iloc[r, :], label.iloc[r, :]

    '''
    选取指定的最大个特征，在这些个特征中，选取分割时的最优特征
    '''
    def select_best_fec(self, data, label):
        fec_num = data.shape[1]
        best_fec_index, best_fec_value = 0, 0
        fec_index = [np.random.randint(fec_num)
                     for i in range(self.max_fec_num)]
        bestS = float('inf')
        S = self.cal_variance(label)
        for index in fec_index:
            for value in set(data.iloc[:, index]):
                l_x, l_y, r_x, r_y = self.split_dataset(
                    data, label, index, value)
                newS = self.cal_variance(l_y) + self.cal_variance(r_y)
                if newS < bestS:
                    bestS = newS
                    best_fec_index = index
                    best_fec_value = value
        if S - bestS < 0.0000001:
            return None, self.cal_mean(label)
        return best_fec_index, best_fec_value

    def build(self, data, label):
        self.tree = self.__build_tree(data, label, 0)
        return self.tree

    def __build_tree(self, data, label, depth):
        best_fec_index, best_fec_value = self.select_best_fec(data, label)
        if best_fec_index == None:
            return best_fec_value
        tree = {}
        if depth >= self.max_depth:
            return self.cal_mean(label)
        tree["best_fec"] = best_fec_index
        tree["best_val"] = best_fec_value
        l_x, l_y, r_x, r_y = self.split_dataset(
            data, label, best_fec_index, best_fec_value)
        tree["left"] = self.__build_tree(l_x, l_y, depth+1)
        tree["right"] = self.__build_tree(r_x, r_y, depth+1)
        return tree

    def predict(self, data):
        if not isinstance(self.tree, dict):
            return None
        return [self.__predict(self.tree, d) for d in data]

    def __predict(self, tree, x):
        if x[tree['best_fec']] > tree['best_val']:
            if type(tree['left']) == float:
                return tree['left']
            return self.__predict(tree['left'], x)
        else:
            if type(tree['right']) == float:
                return tree['right']
            return self.__predict(tree['right'], x)


'''
随机森林需要调整的参数有：
1. 决策树的个数
2. 特征属性的个数
3. 递归次数（即决策树的深度）
'''
class random_forest_regressor:
    def __init__(self, n_estimators=10, max_fec_num=10, max_depth=10):
        self.n_estimators = n_estimators
        self.max_fec_num = max_fec_num
        self.max_depth = max_depth

    '''
    基础实现
    '''
    # 基础实现
    def fit(self, data, label):
        self.trees = []
        for _ in range(self.n_estimators):
            dec_tree = decision_tree_regressor(
                self.max_fec_num, self.max_depth)
            tree = dec_tree.build(data, label)
            self.trees.append(tree)

    # 基础实现
    def predict(self, data):
        if not isinstance(self.trees, list):
            return None
        result = np.zeros(data.shape[0], dtype=np.float)
        for tree in self.trees:
            result += tree.predict(data)
        result /= self.n_estimators
        return result
    
    '''
    并行化
    '''
    # 并行训练
    def fit_worker(self, data, label, q=None):
        dec_tree = decision_tree_regressor(self.max_fec_num, self.max_depth)
        tree = dec_tree.build(data, label)
        if q != None:
            q.put(tree)
        else:
            return tree

    def mul_fit(self, data, label):
        if not isinstance(self.trees, list):
            return None
        q = multiprocessing.Queue()
        jobs = []
        for _ in range(self.n_estimators):
            p = multiprocessing.Process(target=self.fit_worker, args=(data, label, q))
            jobs.append(p)
        for p in jobs:
            p.join()
        self.trees = [q.get() for j in jobs]

    # 并行预测
    def predict_worker(self, tree, data, q=None):
        res = tree.predict(data)
        if q != None:
            q.put(res)
        else:
            return res

    def mul_predict(self, data):
        if not isinstance(self.trees, list):
            return None
        q = multiprocessing.Queue()
        jobs = []
        for tree in self.trees:
            p = multiprocessing.Process(target=self.predict_worker, args=(tree, data, q))
            jobs.append(p)
        for p in jobs:
            p.join()
        result = [q.get() for j in jobs]
        return sum(result) / self.n_estimators

    '''
    进程池并行化
    '''
    def pool_fit(self, data, label):
        if not isinstance(self.trees, list):
            return None
        pool = multiprocessing.Pool(processes=4)
        self.trees = []
        jobs = []
        for _ in range(self.n_estimators):
            p = pool.apply_async(self.fit_worker, (data, label, ))
            jobs.append(p)
        pool.close()
        pool.join()
        self.trees = [j.get() for j in jobs]

    def pool_predict(self, data):
        if not isinstance(self.trees, list):
            return None
        pool = multiprocessing.Pool(processes=4)
        jobs = []
        for tree in self.trees:
            p = pool.apply_async(self.predict_worker, (tree, data, ))
            jobs.append(p)
        pool.close()
        pool.join()
        result = [j.get() for j in jobs]
        return sum(result) / self.n_estimators


def main():
    print("Read Data")
    data_train_1 = pd.read_csv("./data/train1.csv", header=None)
    data_train_2 = pd.read_csv("./data/train2.csv", header=None)
    data_train_3 = pd.read_csv("./data/train3.csv", header=None)
    data_train_4 = pd.read_csv("./data/train4.csv", header=None)
    data_train_5 = pd.read_csv("./data/train5.csv", header=None)

    data_test_1 = pd.read_csv("./data/test1.csv", header=None)
    data_test_2 = pd.read_csv("./data/test2.csv", header=None)
    data_test_3 = pd.read_csv("./data/test3.csv", header=None)
    data_test_4 = pd.read_csv("./data/test4.csv", header=None)
    data_test_5 = pd.read_csv("./data/test5.csv", header=None)
    data_test_6 = pd.read_csv("./data/test6.csv", header=None)

    label_1 = pd.read_csv("./data/label1.csv", header=None)
    label_2 = pd.read_csv("./data/label2.csv", header=None)
    label_3 = pd.read_csv("./data/label3.csv", header=None)
    label_4 = pd.read_csv("./data/label4.csv", header=None)
    label_5 = pd.read_csv("./data/label5.csv", header=None)

    x = pd.concat([data_train_1, data_train_2, data_train_3, data_train_4, data_train_5], ignore_index=True)
    y = pd.concat([label_1, label_2, label_3, label_4, label_5], ignore_index=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    print("train fit")
    random_forest = random_forest_regressor(8, 5, 8)
    # random_forest.fit(x_train, y_train)
    random_forest.mul_fit(x_train, y_train)


    print("train predict")
    # predictions = random_forest.predict(x_test)
    predictions = random_forest.mul_predict(x_test)
    score = r2_score(y_test, predictions)
    print("R2 Score: %.2f%%" % (score * 100.0))


    print("test")
    test = pd.concat([data_test_1, data_test_2, data_test_3, data_test_4, data_test_5, data_test_6], ignore_index=True)
    test_result = random_forest.predict(test)
    pd.DataFrame(data={"Id": [i for i in range(1, len(
        test) + 1)], "Predicted": test_result}).to_csv('result/rf.csv', index=False)

if __name__ == "__main__":
    main()
