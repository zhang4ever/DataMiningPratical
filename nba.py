#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @File    : nba.py
# @Time    : 2017-07-07 14:42
# @Author  : zhang bo
# @Note    : 使用决策树进行nba预测
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

'''数据加载以及数据清洗'''
data_file = 'D:\PythonProjects\PythonDataMining\data\sportsref_download.csv'
data_set = pd.read_csv(data_file, parse_dates=['Date'])[:1230]  # 将第一列变为Date格式
data_set.columns = ['Date', 'Start Time', 'Visitor Team', 'VisitorPts', 'Home Team',
                    'HomePts', 'Score Type', 'OT?', 'Notes']  # rename header name

'''提取新特征：两队各自上一场的输赢情况'''
data_set['HomeWin'] = data_set['VisitorPts'] < data_set['HomePts']  # home win
print("Home Win percentage: {0:.1f}%".format(100 * data_set["HomeWin"].sum() / data_set["HomeWin"].count()))
y_true = data_set['HomeWin'].values

data_set['HomeLastWin'] = False
data_set['VisitorLastWin'] = False

won_last = defaultdict(int)  # 记录上场比赛是否获胜
for index, row in data_set.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    row['HomeLastWin'] = won_last[home_team]
    row['VisitorLastWin'] = won_last[visitor_team]
    data_set.ix[index] = row
    won_last[home_team] = row['HomeWin']
    won_last[visitor_team] = not row['HomeWin']

# 使用决策树
clf = DecisionTreeClassifier(random_state=14)
X_previouswins = data_set[['HomeLastWin', 'VisitorLastWin']].values
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print('Accuracy:{0: .1f}%'.format(np.mean(scores)*100))

'''提取特征:球队排名'''
standing_file = 'D:\PythonProjects\PythonDataMining\data\standings.csv'
standings = pd.read_csv(standing_file, skiprows=[0])
# 查找主队和客队的战绩排名
data_set['HomeTeamRankHigher'] = 0  # new feature
for index, row in data_set.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    if home_team == 'New Orleans Pelicans':  # 球队换名字了
        home_team = 'New Orleans Hornets'
    elif visitor_team == 'New Orleans Pelicans':
        visitor_team = 'New Orleans Hornets'
    home_rank = standings[standings['Team'] == home_team]['Rk'].values[0]  # 主队战绩
    visitor_rank = standings[standings['Team'] == visitor_team]['Rk'].values[0]  # 客队战绩
    row['HomeTeamRankHigher'] = int(home_rank > visitor_rank)  # 更新特征值
    data_set.ix[index] = row

# 改进后的决策树accu= 60.3
X_homehigher = data_set[['HomeLastWin', 'VisitorLastWin', 'HomeTeamRankHigher']].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


'''继续提取新的特征：两队上一场对决胜负: accu=60.6'''
last_match_winner = defaultdict(int)
data_set['HomeTeamWonLast'] = 0  # 主队上次获胜
for index, row in data_set.iterrows():
    home_team = row['Home Team']
    visitor_team = row['Visitor Team']
    terms = tuple(sorted([home_team, visitor_team]))
    row['HomeTeamWonLast'] = 1 if last_match_winner[terms] == row['Home Team'] else 0
    data_set.ix[index] = row
    winner = row['Home Team'] if row['HomeWin'] else row['Visitor Team']  # 更新上那次对决的胜者
    last_match_winner[terms] = winner

X_lastwinner = data_set[['HomeTeamRankHigher', 'HomeTeamWonLast']].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_lastwinner, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


'''主队名字进行数值化。离散化：accu=60.3'''
encoding = LabelEncoder()
encoding.fit(data_set['Home Team'].values)  # 将主队名称转化成整数
home_team = encoding.transform(data_set['Home Team'].values)
visitor_team = encoding.transform(data_set['Visitor Team'])
X_terms = np.vstack([home_team, visitor_team]).T
# 以上述的转化会被视作连续的变量，所以将其转换成二进制来处理
onehot = OneHotEncoder()
X_terms_expanded = onehot.fit_transform(X_terms).todense()
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_terms_expanded, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))


'''使用随机森林accu=60.9'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_terms_expanded, y_true, scoring='accuracy')
print('Accuracy:{0: 0.1f}%'.format(np.mean(scores) * 100))

# 使用不同的特征进行学习accu=61.9
X_all = np.hstack([X_homehigher, X_terms_expanded])
clf = RandomForestClassifier(random_state=14)
scores = cross_val_score(clf, X_all, y_true, scoring='accuracy')
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

'''自动调参'''
X_all = np.hstack([X_homehigher, X_terms])
# 使用GridSearchCV搜索最佳的模型参数
parameters_space = {  # 需要训练的参数
    'max_features': [2,10, 'auto'],
    'n_estimators': [100, ],
    'criterion': ['gini', 'entropy'],
    'min_samples_leaf': [2, 4, 6]
}
clf = RandomForestClassifier(random_state=14)
grid = GridSearchCV(clf, parameters_space)
grid.fit(X_all, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))


'''继续创建新特征'''
data_set['New Feature'] = feat