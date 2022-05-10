from typing import Any
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template, session
import io
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import base64
from sklearn.metrics import roc_curve, auc


app = Flask(__name__)

df_train = pd.read_csv(
    'exo_csv/exoTrain.csv', low_memory=False)
df_test = pd.read_csv(
    'exo_csv/exoTest.csv', low_memory=False)

train_exo_y = df_train[df_train['LABEL'] > 1]
train_exo_n = df_train[df_train['LABEL'] < 2]
train_t_n = train_exo_n.iloc[:, 1:].T
train_t_y = train_exo_y.iloc[:, 1:].T

# La colonne label est la seul colonne de catégorie dans le dataset !
X_train, y_train = df_train.drop(columns=['LABEL'], axis=1), df_train['LABEL']
X_test, y_test = df_test.drop(columns=['LABEL'], axis=1), df_test['LABEL']


X_test_1, X_test_2 = X_test[:285], X_test[285:]
y_test_1, y_test_2 = y_test[:285], y_test[285:]

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_1_scaled = scaler.transform(X_test_1)
X_test_2_scaled = scaler.transform(X_test_2)


@app.route('/', methods=("POST", "GET"))
def html_table():

    return render_template('index.html', tables=[df_train.head().to_html()], titles=df_train.columns.values)


@app.route('/plot')
# graph
def graph():
    fig = plt.figure(figsize=(4, 6))
    colors = ["0", "1"]
    sns.countplot('LABEL', data=df_train, palette="Set2")
    plt.title('Distribution des classes \n (0: Non-Exoplanete || 1: Exoplanete)', fontsize=14)
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("visu.html", image=pngImageB64String)


@app.route('/plot2')
# graph2
def graph2():
    fig2, ax2 = plt.subplots()
    X_train.loc[y_train == 2][:5].T.plot(subplots=True, figsize=(10, 12), ax=ax2, sharex=True, sharey=False, legend=False)
    img = io.BytesIO()
    FigureCanvas(fig2).print_png(img)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("visu2.html", image2=pngImageB64String)


@app.route('/plot3')
# graph2
def graph3():
    fig3, ax3 = plt.subplots()
    X_train.loc[y_train == 1][:5].T.plot(subplots=True, figsize=(10, 12), ax = ax3, sharex = True, sharey = False, legend = False)
    img = io.BytesIO()
    FigureCanvas(fig3).print_png(img)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("visu3.html", image3=pngImageB64String)

@app.route('/forest')
def html_forest():

    RF = RandomForestClassifier()
    RF.fit(X_train_scaled, df_train['LABEL'])

    score_trainRF = RF.score(X_train_scaled, df_train['LABEL'])
    score_test1RF = RF.score(X_test_1_scaled, y_test_1)
    score_test2RF = RF.score(X_test_2_scaled, y_test_2)
    
    return render_template("forest.html", score1=score_trainRF, score2=score_test1RF, score3=score_test2RF)


@app.route('/log')
def html_log():

    LR = LogisticRegression()
    LR.fit(X_train_scaled, df_train['LABEL'])
    prediction = LR.predict(X_test)

    score_trainLR = LR.score(X_train_scaled, df_train['LABEL'])
    score_test1LR = LR.score(X_test_1_scaled, y_test_1)
    score_test2LR = LR.score(X_test_2_scaled, y_test_2)

    return render_template("log.html", score1=score_trainLR, score2=score_test1LR, score3=score_test2LR)



# GradientBoostingClassifier
@app.route('/grad')
def html_grad():
    GB = GradientBoostingClassifier()
    GB.fit(X_train_scaled, df_train['LABEL'])

    score_trainGB = GB.score(X_train_scaled, df_train['LABEL'])
    score_test1GB = GB.score(X_test_1_scaled, y_test_1)
    score_test2GB = GB.score(X_test_2_scaled, y_test_2)

    return render_template("log.html", score1=score_trainGB, score2=score_test1GB, score3=score_test2GB)



if __name__ == '__main__':
    app.run()
