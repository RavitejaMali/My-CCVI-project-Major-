# -*- coding: utf-8 -*-
"""
Updated on Sat Apr 15 11:49:48 2023
"""

from flask import Flask, request, render_template
from flask_cors import cross_origin
from scipy.io import loadmat
import sklearn
import pickle
import pandas as pd
import os
import base64
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import webbrowser
import boto3


def plot_band(dataset):
    plt.figure(figsize=(8, 6))
    band_no = np.random.randint(dataset.shape[2])
    plt.imshow(dataset[:,:, band_no], cmap='jet')
    plt.title(f'Band-{band_no}', fontsize=14)
    plt.axis('off')
    plt.colorbar()
    #plt.plot(dataset[:,:, band_no], cmap='jet')
    #plt.show()
    plt.savefig(OUTPUT_FOLDER + "/band.png")   
    return band_no

def plot_rgb(Red_B4, Green_B3, Blue_B2, NIR_B5):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 14))
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax3 = axes[0, 1]
    ax4 = axes[1, 1]

    # Plot Red, Green, Blue and NIR

    # show(raster_red, cmap='Reds',  ax=ax1)
    # max_val_red = np.max(red_np) ## maximum value of the red reflectance 
    img = ax1.imshow(Red_B4, cmap = 'Reds')
    img.set_clim(vmin=0, vmax=5000) ## set the maximum of the colormap to get better visualisation
    fig.colorbar(img, ax = ax1)

    ## Green
    img = ax2.imshow(Green_B3, cmap = 'Greens')
    img.set_clim(vmin=0, vmax=3000)
    fig.colorbar(img, ax = ax2)

    ## Blue
    img = ax3.imshow(Blue_B2, cmap = 'Blues')
    img.set_clim(vmin=0, vmax=3000)
    fig.colorbar(img, ax = ax3)

    ### NIR4
    img = ax4.imshow(NIR_B5)
    img.set_clim(vmin=3000, vmax=5000)
    fig.colorbar(img, ax = ax4)

    # Add titles
    ax1.set_title("Red")
    ax2.set_title("Green")
    ax3.set_title("Blue")
    ax4.set_title("Nir")
    #fig.savefig(OUTPUT_FOLDER + "/band_RGB.png")

def vege_indices(NIR_B8, Red_B4, Green_B3, NIR_B6, NIR_B5, NIR_B8A, SWIR_B11):
    ### Normalized Difference Vegetation Index  (NDVI)
    def ndvi(B08, B04):
        index = (B08 - B04) / (B08 + B04)
        return index

    # Green Normalized Difference Vegetation Index   (GNDVI)
    def gndvi(B08, B03):
        index = (B08 - B03) / (B08 + B03)
        return index

    # Chlorophyll Red-Edge
    def red_edge(B07, B05):
        #index = math.pow((B07 / B05), (-1.0))
        index = (B07 / B05)**-1
        return index

    # Normalized Difference 819/1600 NDII (NDII)
    def ndii(B8A, B11):
        index = (B8A - B11) / (B8A + B11)
        return index

    # MSI - Simple Ratio 1600/820 Moisture Stress Index (MSI)
    def msi(B11, B8A):
        index = B11 / B8A
        return index

    # NDVI 
    out_ndvi = ndvi(NIR_B8, Red_B4)
    #ndvi_2 = out_ndvi * 100
    #ndvi_2[ndvi_2 < 0] = 111
    fig, ax = plt.subplots(figsize = (15, 12))
    img = ax.imshow(out_ndvi, cmap = 'PiYG')
    # img.set_clim(vmin=20, vmax=70)
    fig.colorbar(img, ax = ax)
    #fig.savefig(OUTPUT_FOLDER + "/NDVI_Indices.png")

    # GNDVI
    out_gndvi = gndvi(NIR_B8, Green_B3)

    fig, ax = plt.subplots(figsize = (15, 12))
    img = ax.imshow(out_gndvi, cmap = 'PiYG')
    # img.set_clim(vmin=20, vmax=70)
    fig.colorbar(img, ax = ax)
    #fig.savefig(OUTPUT_FOLDER + "/GNDVI_Indices.png")

    out_redge = red_edge(NIR_B6, NIR_B5)

    fig, ax = plt.subplots(figsize = (15, 12))
    img = ax.imshow(out_redge, cmap = 'PiYG')
    # img.set_clim(vmin=20, vmax=70)
    fig.colorbar(img, ax = ax)
    #fig.savefig(OUTPUT_FOLDER + "/Chorophyll_Indices.png")

    out_ndii = ndii(NIR_B8A, SWIR_B11)

    fig, ax = plt.subplots(figsize = (15, 12))
    img = ax.imshow(out_ndii, cmap = 'PiYG')
    # img.set_clim(vmin=20, vmax=70)
    fig.colorbar(img, ax = ax)
    #fig.savefig(OUTPUT_FOLDER + "/NDII_Indices.png")

    out_msi = msi(SWIR_B11, NIR_B8A)

    fig, ax = plt.subplots(figsize = (15, 12))
    img = ax.imshow(out_msi, cmap = 'PiYG')
    # img.set_clim(vmin=20, vmax=70)
    fig.colorbar(img, ax = ax)
    #fig.savefig(OUTPUT_FOLDER + "/MSI_Indices.png")
'''# Actual Code
def plot_signature(df):
    
    plt.figure(figsize=(12, 6))
    pixel_no = np.random.randint(df.shape[0])
    plt.plot(range(1, 201), df.iloc[pixel_no, :-1].values.tolist(), 'b--', label= f'Class - {df.iloc[pixel_no, -1]}')
    plt.legend()
    plt.title(f'Pixel({pixel_no}) signature', fontsize=14)
    plt.xlabel('Band Number', fontsize=14)
    plt.ylabel('Pixel Intensity', fontsize=14)
    plt.savefig(OUTPUT_FOLDER + "/Spectral_plot.png")
    return pixel_no
'''

# trial code
def plot_signature(df):
    plt.figure(figsize=(12, 6))
    pixel_no = np.random.randint(df.shape[0])
    n_bands = df.shape[1] - 1
    plt.plot(range(1, n_bands+1), df.iloc[pixel_no, :-1].values.tolist(), 'b--', label= f'Class - {df.iloc[pixel_no, -1]}')
    plt.legend()
    plt.title(f'Pixel({pixel_no}) signature', fontsize=14)
    plt.xlabel('Band Number', fontsize=14)
    plt.ylabel('Pixel Intensity', fontsize=14)
    plt.savefig(OUTPUT_FOLDER + "/Spectral_plot.png")
    return pixel_no


current_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join('data')
app = Flask(__name__,  template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

OUTPUT_FOLDER=os.path.join(os.path.expanduser('~'),"Downloads")


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html") 

@app.route("/preprocessed", methods = ["GET", "POST"])
@cross_origin()
def preprocessed():
    if request.method == "POST":
        # Date_of_Journey
        file_path_dt = request.form["myfile_dt"]
        train = os.path.join(app.config['UPLOAD_FOLDER'], file_path_dt)
        file_path_gt = request.form["myfile_gt"]
        valid = os.path.join(app.config['UPLOAD_FOLDER'], file_path_gt)
        
        if "submit_button_1" in request.form.keys():
            
#        if request.form["submit_button_1"] == "Submit_1":
            #train_df_shape, valid_df_shape, test_df_shape, plot, train_df_final = main(train, valid, test)
            
            
            #dataset = loadmat(train)['salinas_corrected']  
            #ground_truth = loadmat(valid)['salinas_gt']
            
            # trail code here
            dataset = loadmat(train)[os.path.splitext(file_path_dt)[0]]
            ground_truth = loadmat(valid)[os.path.splitext(file_path_gt)[0]]
    
            
            
            #dataset = loadmat(train)['indian_pines_corrected']          
            #ground_truth = loadmat(valid)['indian_pines_gt']
            a = dataset.shape
            b = ground_truth.shape
            print(ground_truth.shape[0])
            print(f'Dataset: {a}\nGround Truth: {b}')
            a1 = plot_band(dataset)
            c = 'Visualization of Band ' + str(a1) + ' is saved in Output Folder'

            Red_B4 = dataset[:,:, 29]
            Green_B3 = dataset[:,:, 20]
            Blue_B2 = dataset[:,:, 12]
            NIR_B5 = dataset[:,:, 39]
            NIR_B6 = dataset[:,:, 42]
            NIR_B8 = dataset[:,:, 53]
            NIR_B8A = dataset[:,:, 54]
            SWIR_B11 = dataset[:,:, 121]

            plot_rgb(Red_B4, Green_B3, Blue_B2, NIR_B5)
            d = 'Visualization of RGB Band is saved in Output Folder'

            i = 'Vegetation Indices Processing Started'
            vege_indices(NIR_B8, Red_B4, Green_B3, NIR_B6, NIR_B5, NIR_B8A, SWIR_B11)
            j = 'Plot of NDVI is saved in Output Folder'
            k = 'Plot of GNDVI is saved in Output Folder'
            l = 'Plot of Chlorophyll Red Edge is saved in Output Folder'
            m = 'Plot of NDII is saved in Output Folder'
            n = 'Plot of MSI is saved in Output Folder'
            o = 'Vegetation Indices Processing is Completed'

            def extract_pixels(dataset, ground_truth):
                df = pd.DataFrame()
                for i in tqdm(range(dataset.shape[2])):
                    df = pd.concat([df, pd.DataFrame(dataset[:, :, i].ravel())], axis=1)
                df = pd.concat([df, pd.DataFrame(ground_truth.ravel())], axis=1)
                df.columns = [f'band-{i}' for i in range(1, 1+dataset.shape[2])]+['class']
                return df




            df = extract_pixels(dataset, ground_truth)
            a2 = plot_signature(df)
            p = 'Spectral Plot of Pixel ' + str(a2) + ' is saved in Output Folder'

            print(f'Shape of the data: {df.shape}')
            print(f"Unique Class Labels: {df.loc[:, 'class'].unique()}")
            df.loc[:, 'class'].value_counts()

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            print(X.shape)
            print(y.shape)
            #Actual Code
            '''
            pca = PCA(n_components = 150)
            principalComponents = pca.fit_transform(X)
            ev=pca.explained_variance_ratio_
            '''           
            # trail code 1
            
            n_components = dataset.shape[2] - 50
            pca = PCA(n_components=n_components)
            principalComponents = pca.fit_transform(X)
            ev = pca.explained_variance_ratio_
            
            print('Test')
            def plot_pca(ev):
                plt.plot(np.cumsum(ev))
                plt.xlabel('Number of components')
                plt.ylabel('Cumulative explained variance')
                plt.savefig(OUTPUT_FOLDER + "/output/PCA_plot.png")

            q = 'Plot of PCA Component is saved in Output Folder'
            X_train, X_test, y_train, y_test, indices_train, indices_test  = train_test_split(principalComponents, y,  range(X.shape[0]), 
                                                                                  test_size = 0.15, random_state = 11)
            
            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

            svm = SVC(kernel='rbf', degree = 10, gamma='scale', cache_size=1024*7)
            print('Test 1')
            svm.fit(X_train, y_train)
            print('Test 2')
            y_pred = svm.predict(X_test)
            r = 'Crop Classification Model is Started'
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            print(f'Accuracy: {accuracy_score(y_test, y_pred)}%')
            acc = accuracy_score(y_test, y_pred) * 100
            s = 'Accuracy :- '+str(acc) + '%'
            skplt.metrics.plot_confusion_matrix(
                y_test, 
                y_pred,
                figsize=(12,12));
            print('Test 3')
            t = 'Confusion Matrix Plot is saved in Output Folder'
            classi = classification_report(y_test,y_pred, output_dict=True)
            print('Classification report:\n',classi)
            df = pd.DataFrame(classi)
            df.to_csv(OUTPUT_FOLDER + '/Classification Report.csv', index=True)
            u = 'Classification Report is saved in output folder'
            print('test 4')
            pre = y_pred

            clmap = [0]*X.shape[0]

            for i in tqdm(range(len(indices_train))):
                clmap[indices_train[i]] = y[indices_train[i]]

            for i in tqdm(range(len(indices_test))):
                clmap[indices_test[i]] = pre[i]
            print('test 5')
            # Actual Code 
            
            def prdict(clmap):
                plt.figure(figsize=(8, 6))
                plt.imshow(np.array(clmap).reshape((ground_truth.shape[0], ground_truth.shape[1])), cmap='jet')
                plt.colorbar()
                plt.axis('off')
                plt.title('Classification Map (PCA + SVM)')
                plt.savefig(OUTPUT_FOLDER + "/Predicted_image.png")
                #plt.show()
            
            
            # Trail code 
            '''def prdict(clmap):
               clmap_arr = np.array(clmap)
               height, width = clmap_arr.shape[:2]
               plt.figure(figsize=(8, 6))
               plt.imshow(clmap_arr, cmap='jet')
               plt.colorbar()
               plt.axis('off')
               plt.title('Classification Map (PCA + SVM)')
               plt.savefig(UPLOAD_FOLDER + "/output/Predicted_image.png")
               '''
              

                             
            prdict(clmap)
            
            x = 'Predicted Satellite Image is saved in output Folder'
            y = 'Crop Classification is Completed '
            return render_template('preprocessing.html',a=a, b=b, c=c, d=d, i=i, j=j, k=k, l=l, m=m, n=n, o=o, p=p, q=q, r=r, s=s, t=t, u=u, x=x, y=y)
            #return a

        return render_template("home.html")
    

if __name__ == "__main__":
    #url="http://localhost:5000/"
    #url = "https://www.google.com"
    #webbrowser.open(url)
    app.run(host="localhost", port=5000, debug=False) 
    
    #webbrowser.open(url)
    
