import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
#S
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    
    val = to_excel(df)
    b64 = base64.b64encode(val) 
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download File</a>'


def get_sex(n):
	if n=="Male":
		return 1
	else:
		return 0
def get_smoking(n):
	if n=="Yes":
		return 1
	else:
		return 0

def get_age(n):
	return n**(1/2)
def get_chol(n):
	return n**(1/2)
def get_bmi(n):
	return n**(1/2)
def get_cp(n):
	if n=='0':
		return 0.0
	elif n=='1':
		return 1.0
	elif n=='2':
		return 2.0
	elif n=='3':
		return 3.0
def get_exg(n):
	if n=='0':
		return 0.0
	elif n=='1':
		return 1.0

	elif n=='2':
		return 2.0
def get_LAD(n):
	if n=='10-20':
		return 0
	elif n=='20-30':
		return 1
	elif n=='30-40':
		return 2
	elif n=='40-50':
		return 3
	elif n=='50-60':
		return 4
	elif n=='60-70':
		return 5
	elif n=='70-80':
		return 6
	elif n=='80-90':
		return 7
	elif n=='90-100':
		return 9
	else: 
		return 8
def get_CA(n):
	if n=='10-20':
		return 0
	elif n=='20-30':
		return 1
	elif n=='30-40':
		return 2
	elif n=='40-50':
		return 3
	elif n=='50-60':
		return 4
	elif n=='60-70':
		return 5
	elif n=='70-80':
		return 6
	elif n=='80-90':
		return 7
	elif n=='90-100':
		return 9
	else: 
		return 8
def get_RCA(n):
	if n=='10-20':
		return 0
	elif n=='20-30':
		return 1
	elif n=='30-40':
		return 2
	elif n=='40-50':
		return 3
	elif n=='50-60':
		return 4
	elif n=='60-70':
		return 5
	elif n=='70-80':
		return 6
	elif n=='80-90':
		return 7
	elif n=='90-100':
		return 9
	else: 
		return 8

def get_OM(n):
	if n=='10-20':
		return 0**(1/2)
	elif n=='20-30':
		return 1**(1/2)
	elif n=='30-40':
		return 2**(1/2)
	elif n=='40-50':
		return 3**(1/2)
	elif n=='50-60':
		return 5**(1/2)
	elif n=='60-70':
		return 6**(1/2)
	elif n=='70-80':
		return 7**(1/2)
	elif n=='80-90':
		return 8**(1/2)
	elif n=='90-100':
		return 10**(1/2)
	else: 
		return 9**(1/2)
def get_hrt(n):
	return n**(1/2)
def get_pulse(n):
	return n**(1/2)
def get_op(n):
	if n=='1':
		return 1
	else:
		return 0

def main():
	
	#st.title("STENT COUNT PREDICTION")
	st.markdown('<div style=background-color:powderblue;</div>',unsafe_allow_html=True)
	activity=['Home','EDA','Prediction']
	
	st.sidebar.markdown('<div style=color:#006080;font-size:20px;font-family:courier;font-weight:bold>MENU</div>',unsafe_allow_html=True)
	choice=st.sidebar.radio("        ",activity)

	
	img=Image.open('hb3.png')
		
	st.sidebar.image(img,width=300)
	

	if choice=='EDA':
		#st.subheader("Explarotary Data Analysis")
		st.markdown('<div style=color:#001866;text-align:center;font-size:40px;font-family:courier;font-weight:bold>EXPLAROTARY DATA ANALYSIS</div>',unsafe_allow_html=True)
		df=pd.read_csv("dataset - dataset.csv")
		df1=pd.read_csv("stent.csv")
		df2=pd.read_csv("stent1.csv")
		df3=pd.read_csv("Book1.csv")
		st.sidebar.markdown('<div style=color:#006080;font-size:20px;font-family:courier;font-weight:bold>EDA SESSION</div>',unsafe_allow_html=True)
		


		

		
		st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Column Names:</div>',unsafe_allow_html=True)
		st.write(df.columns)
		st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Dataset Description:</div>',unsafe_allow_html=True)
		st.write(df.describe())
		st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Dataset Shape:</div>',unsafe_allow_html=True)
		st.write(df.shape)
		
		#
		#Blues, Blues_r, Pastel2, Pastel2_r
		#st.pyplot(figsize=(10,10))
		st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Graphical Representation Of The Dataset:</div>',unsafe_allow_html=True)
		sns.catplot(x='stent',data=df1,palette='Blues',kind='violin',  height=2, aspect=3)
		#df1['stent'].value_counts().plot(kind='pie', figsize=(20,10),palette='Blues')
		st.pyplot(figsize=(10,10))
		#sns.countplot(x='stent',data=df1,palette='Blues')
		#st.pyplot(figsize=(10,10))
		#st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Graphical Representation To Show The Effect Of Sex On The No Of Stents:</div>',unsafe_allow_html=True)
		#sns.countplot(x='sex',hue='stent',data=df1,palette='Blues')
		#st.pyplot(figsize=(10,10))
		#3st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Graphical Representation To Show The Effect Of Smoking On The No Of Stents:</div>',unsafe_allow_html=True)
		#sns.countplot(x='smoking',hue='stent',data=df1,palette='Blues')
		#st.pyplot(figsize=(10,10))
		if st.sidebar.checkbox("View Dataset"):
			n=st.sidebar.number_input(" Select Number of Rows",value=1)
			st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Dataset:</div>',unsafe_allow_html=True)
			
			st.dataframe(df.head(n))
			#st.markdown(get_table_download_link(df.head(number)), unsafe_allow_html=True)


		
		if st.sidebar.checkbox("View Selected Columns"):
			all_columns=df.columns.tolist()
			selected_columns=st.sidebar.multiselect("Select Columns",all_columns)
			new_df=df[selected_columns]
			st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Selected Columns:</div>',unsafe_allow_html=True)

		
		
			
			st.dataframe(new_df)

		if st.sidebar.checkbox("View Selected Rows"):
			st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Selected Rows:</div>',unsafe_allow_html=True)
			selected_index=st.sidebar.multiselect("Select rows",df.index)
			selected_rows=df.loc[selected_index]
		#if st.button("View Selected Rows"):
			
			st.dataframe(selected_rows)	

		if st.sidebar.checkbox("View Test Data"):
			number=st.sidebar.number_input("Number of Rows",value=1)
			st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Test Data:</div>',unsafe_allow_html=True)
			
			st.dataframe(df2.head(number))
			st.markdown(get_table_download_link(df2.head(number)), unsafe_allow_html=True)


		if st.sidebar.checkbox("View Executed Data"):
			#number=st.sidebar.number_input("Number of Rows",value=1)
			st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Executed Data:</div>',unsafe_allow_html=True)
			
			st.dataframe(df3)
			st.markdown(get_table_download_link(df3), unsafe_allow_html=True)






		


	if choice=='Prediction':
		df1=pd.read_csv("stent.csv")

		st.sidebar.markdown('<div style=color:#006080;font-size:20px;font-family:courier;font-weight:bold>ENTER PATIENT DETAILS:</div>',unsafe_allow_html=True)
		st.markdown('<div style=color:#001866;text-align:center;font-size:40px;font-family:courier;font-weight:bold>PREDICTION SESSION</div>',unsafe_allow_html=True)
		st.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Graphical Representation Of The Dataset:</div>',unsafe_allow_html=True)
		#sns.catplot(x='stent',data=df1,palette='Blues',kind='violin',  height=2, aspect=3)
		#df1['stent'].value_counts().plot(kind='pie', figsize=(20,10),palette='Blues')
		#st.pyplot(figsize=(5,5))
		img=Image.open('cg.png')
		
		st.image(img,width=700)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Sex:</div>',unsafe_allow_html=True)
		sex=st.sidebar.radio(' ',['Male','Female'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Age:</div>',unsafe_allow_html=True)
		age=st.sidebar.number_input(" ",value=1)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Smoking:</div>',unsafe_allow_html=True)
		smk=st.sidebar.radio('  ',['Yes','No'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Cholestrol:</div>',unsafe_allow_html=True)
		chol=st.sidebar.number_input(" ")
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Chest Pain:</div>',unsafe_allow_html=True)
		cp=st.sidebar.radio('   ',['0','1','2','3'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Maximum Heart Rate:</div>',unsafe_allow_html=True)
		hrt=st.sidebar.number_input("  ")
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Body Mass Index:</div>',unsafe_allow_html=True)
		bmi=st.sidebar.number_input("   ")
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>ST Depression Induced By Exercise Relative To Rest:</div>',unsafe_allow_html=True)
		op=st.sidebar.radio('    ',['0','1'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Slope Of The Peak Exercise ST Segment:</div>',unsafe_allow_html=True)
		slp=st.sidebar.number_input("    ")
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Exercise Induced Angina:</div>',unsafe_allow_html=True)

		exg=st.sidebar.radio("     ",["0","1","2"],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Percentage Of Blockage In The Left Anterior Descending Artery:</div>',unsafe_allow_html=True)
		lad=st.sidebar.selectbox(' ',['0','1-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Percentage Of Blockage In The Obtuse Marginal Artery:</div>',unsafe_allow_html=True)
		om=st.sidebar.selectbox('  ',['0','1-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Percentage Of Blockage In The Circumflex Artery:</div>',unsafe_allow_html=True)
		ca=st.sidebar.selectbox('   ',['0','1-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],index=0)
		st.sidebar.markdown('<div style=color:#3973ac;font-size:16px;font-weight:bold;font-family:courier>Percentage Of Blockage In The Right Coronary Artery:</div>',unsafe_allow_html=True)
		rca=st.sidebar.selectbox('     ',['0','1-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'],index=0)
		sex1=get_sex(sex)
		age1=get_age(age)
		smk1=get_smoking(smk)
		chol1=get_chol(chol)
		cp1=get_cp(cp)
		hrt1=get_hrt(hrt)
		bmi1=get_bmi(bmi)
		op1=get_op(op)
		slp1=slp
		lad1=get_LAD(lad)
		om1=get_OM(om)
		rca1=get_RCA(rca)
		ca1=get_CA(ca)
		exg1=get_exg(exg)
		lst=[sex1,age1,smk1,chol1,cp1,hrt1,bmi1,op1,slp1,exg1,lad1,om1,ca1,rca1]
		#st.write(lst)
		#img=Image.open('hb5.jpg')
		
		#st.image(img,width=300)

		sample_data=np.array(lst).reshape(1,-1)
		#st.write(sample_data)
		if st.sidebar.button("Predict"):
			#st.markdown('<div style=color:black;font-size:22px;font-family:courier>Data Encrypted As:</div>',unsafe_allow_html=True)
			#st.write(lst)
			model=joblib.load(open(os.path.join("dtree_model.pkl"),"rb"))
			Prediction=model.predict(sample_data)
			if Prediction==0.:
				st.markdown('<div style=color:black;font-size:22px;font-family:courier>Number of stents required:0</div>',unsafe_allow_html=True)
			if Prediction==1.:
				st.markdown('<div style=color:black;font-size:22px;font-family:courier>Number of stents required:1</div>',unsafe_allow_html=True)
			if Prediction==2.:
				st.markdown('<div style=color:black;font-size:22px;font-family:courier>Number of stents required:2</div>',unsafe_allow_html=True)
			if Prediction==3.:
				st.markdown('<div style=color:black;font-size:22px;font-family:courier>Number of stents required:3</div>',unsafe_allow_html=True)
			img=Image.open('hb3.png')
		
			st.image(img,width=700)


	if choice=='Home':
		st.markdown('<div style=color:#001866;text-align:center;font-size:40px;font-family:courier;font-weight:bold>STENT COUNT PREDICTION</div>',unsafe_allow_html=True)
		img=Image.open('rb.jpg')
		
		st.image(img,width=120)
		st.markdown('<div style=color:black;font-size:16px;font-family:times>A little prediction goes a long way and this is what we tried implementing Where data is the new science, Machine Learning holds all the answers.With all data piled up, Machine Learning accomplishes the task of developing new capabilities from these data.This automation is the source of our model for building its predictive power</div>',unsafe_allow_html=True)
		st.write(" ")
		img=Image.open('cg.png')
		
		st.image(img,width=700)
		st.markdown('<div style=color:black;font-size:16px;font-family:times>In this app,in order to predict the risk of angioplasty, user needs to enter the values of various parameters on the basis of which the calculation will be made. From the given measurements,our app will evaluate the number of stents required by the patient.</div>',unsafe_allow_html=True)
		st.write(" ")
		img=Image.open('hb3.png')
		
		st.image(img,width=700)



	st.sidebar.markdown('<div style=color:#006080;font-size:20px;font-family:courier;font-weight:bold>ABOUT US</div>',unsafe_allow_html=True)
	st.sidebar.markdown('<div style=color:black;font-size:12px;font-family:times>Every year a large amount of data is generated in the healthcare industry but they are not used effectively.So here is a system that will be able to communicate with people’s mind and help them in getting prepared based on their medical history.By using patient’s input features such as sex, cholesterol, blood pressure, smoking habits, BMI and much more the prediction of their angioplasty can be done. So this model depicts whether or not a person needs stent implantation and if needs be then what would be the number of stents required.</div>',unsafe_allow_html=True)
	#img=Image.open('bulb.png')
	#img=Image.open('bub.png')
		
	#st.sidebar.image(img,width=300)
	#img=Image.open('cg.png')
		
	#st.image(img,width=700)

	Image.open('bulb3.jpg').convert('RGB').save('new2.png')	
	st.sidebar.image('new2.png',width=40)

	st.sidebar.markdown('<div style=color:black;font-size:12px;font-family:courier>Use the values from the dataset to see different results.</div>',unsafe_allow_html=True)

	


if __name__ == '__main__':
	main()
