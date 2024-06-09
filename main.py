from flask import Flask, render_template, request, redirect
from flask import request
import pandas as pd
from lib.drop import dropcolumn
from lib.kmeans import kmeans_and_logistic_regression
from lib.kmedian import k_median_clustering

app = Flask(__name__)

@app.route('/')
def index():
    model = request.args.get('model')
    file = request.args.get('file')
    nilaik = request.args.get('nilaik')
    if (model != None) and (file != None) and (nilaik != None):
        datacsv = pd.read_csv(file)
        header = datacsv.columns
        print(header)
        body = datacsv.values
        return render_template('step2.html', header=header, body=body, model=model, file=file, nilaik=nilaik)
    else:
        return render_template('step1.html')
       
@app.route('/dropcolumn', methods = ['POST'])
def dropcolumn_route():
    model = request.form['model']
    file = request.form['file']
    nilaik = request.form['nilaik']
    dropcolumn_name = request.form['column']
    path = dropcolumn(file, dropcolumn_name)
    return redirect("/?model="+model+"&file="+path+"&nilaik="+nilaik)
        
@app.route('/uploadfile', methods = ['POST'])   
def uploadfile():  
    model = request.form['model']
    file = request.files['file']
    nilaik = request.form['nilaik']
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save("static/csv/" + f.filename)
        path = "static/csv/" + f.filename
        return redirect("/?model="+model+"&file="+path+"&nilaik="+nilaik)

@app.route('/process')
def proses():
    model = request.args.get('model')
    file = request.args.get('file')
    nilaik = request.args.get('nilaik')

    if not model or not file or not nilaik:
        return redirect('/')

    if model == '1':
        datacsv = pd.read_csv(file)
        header = datacsv.columns
        body = datacsv.values

        fitur1 = request.args.get('fitur1')
        fitur2 = request.args.get('fitur2')

        if fitur1 and fitur2:
            kmeans = kmeans_and_logistic_regression(file, int(nilaik), int(fitur1), int(fitur2))
        else:
            kmeans = kmeans_and_logistic_regression(file, int(nilaik))

        return render_template('process.html', akurasi=kmeans['akurasi'], kerekatan=kmeans['kerekatan'], path=kmeans['path'], header=header, body=body, model=model, file=file, nilaik=nilaik)

    elif model == '2':
        datacsv = pd.read_csv(file)
        header = datacsv.columns
        body = datacsv.values

        fitur1 = request.args.get('fitur1')
        fitur2 = request.args.get('fitur2')

        if fitur1 and fitur2:
            kmedian = k_median_clustering(file, int(nilaik), int(fitur1), int(fitur2))
        else:
            kmedian = k_median_clustering(file, int(nilaik))

        return render_template('process.html', akurasi=kmedian['akurasi'], kerekatan=kmedian['kerekatan'], path=kmedian['path'], header=header, body=body, model=model, file=file, nilaik=nilaik)
    else:
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

