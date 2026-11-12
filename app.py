import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from src.model import prepare_data,train_model,predict_one,FEATURE_COLUMNS
st.set_page_config(page_title='FraudShield AI',page_icon='🛡️',layout='wide')
st.markdown('''<style>@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');.stApp{background:radial-gradient(ellipse at 10% 0%,#142b48 0%,#080d18 45%);color:#eaf2ff;font-family:'DM Sans',sans-serif}.block-container{padding-top:1.5rem;max-width:1500px}[data-testid="stSidebar"]{background:#0b1423;border-right:1px solid #263b55}h1,h2,h3{font-family:'Space Grotesk',sans-serif}div[data-testid="stMetric"]{background:linear-gradient(145deg,#14243a,#101a2b);border:1px solid #29415f;padding:15px;border-radius:15px}.hero{padding:24px 28px;border:1px solid #29415f;border-radius:20px;background:linear-gradient(110deg,#173451,#101a2b);margin-bottom:18px}.kicker{color:#36d6e7;font-weight:700;letter-spacing:2px;font-size:.75rem}.hero h1{color:#f5f8ff;margin:7px 0}.hero p{color:#adc0d8;margin:0}.stButton>button{border:1px solid #36d6e7;background:linear-gradient(90deg,#176a83,#2455a5);color:white;border-radius:10px;font-weight:700}</style>''',unsafe_allow_html=True)
def hero(k,t,s): st.markdown(f'<div class="hero"><div class="kicker">{k}</div><h1>{t}</h1><p>{s}</p></div>',unsafe_allow_html=True)
def theme(fig,h=330): fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',height=h,margin=dict(l=12,r=12,t=42,b=12)); return fig
st.sidebar.markdown('# 🛡️ FRAUDSHIELD'); st.sidebar.caption('INTELLIGENT TRANSACTION MONITOR')
page=st.sidebar.radio('NAVIGATION',['Executive Dashboard','Fraud Detection','Batch Analysis','Model Performance','Dataset Explorer','How It Works','About Project'])
up=st.sidebar.file_uploader('Upload Kaggle creditcard.csv',type='csv',key='dataset')
df=None
if up:
    try: df=pd.read_csv(up); st.session_state['df']=df
    except Exception as e: st.sidebar.error(str(e))
elif 'df' in st.session_state: df=st.session_state['df']
elif Path('data/creditcard.csv').exists(): df=pd.read_csv('data/creditcard.csv'); st.session_state['df']=df
model=metrics=None
if df is not None:
    try:
        X,y=prepare_data(df); sig=(len(df),tuple(df.columns))
        if st.session_state.get('sig')!=sig:
            with st.spinner('Training balanced Logistic Regression...'): model,metrics=train_model(X,y)
            st.session_state.update(model=model,metrics=metrics,sig=sig)
        model=st.session_state['model']; metrics=st.session_state['metrics']
    except Exception as e: st.error(f'Dataset setup error: {e}')

def empty(): st.info('Upload the Kaggle creditcard.csv file using the sidebar to activate the model and visualizations.')
if page=='Executive Dashboard':
    hero('FRAUD INTELLIGENCE / OVERVIEW','See the signal. Stop the fraud.','An educational machine-learning workspace for identifying suspicious card transactions.')
    if model is None: empty()
    else:
        total=len(df); fraud=int((df.Class==1).sum()); genuine=total-fraud
        a,b,c,d=st.columns(4); a.metric('TRANSACTIONS',f'{total:,}'); b.metric('FRAUD RECORDS',f'{fraud:,}',f'{fraud/total:.2%} of dataset',delta_color='inverse'); c.metric('GENUINE RECORDS',f'{genuine:,}'); d.metric('MODEL RECALL',f'{metrics["recall"]:.1%}')
        l,r=st.columns(2)
        with l:
            fig=go.Figure(go.Pie(labels=['Genuine','Fraud'],values=[genuine,fraud],hole=.68,marker_colors=['#3988ff','#ff5575'])); fig.update_layout(annotations=[dict(text=f'{total:,}<br>records',x=.5,y=.5,showarrow=False)]); st.plotly_chart(theme(fig),use_container_width=True)
        with r:
            sm=df.sample(min(12000,len(df)),random_state=7); sm=sm.assign(Type=sm.Class.map({0:'Genuine',1:'Fraud'})); st.plotly_chart(theme(px.histogram(sm,x='Amount',color='Type',nbins=55,barmode='overlay',opacity=.7,color_discrete_map={'Genuine':'#3988ff','Fraud':'#ff5575'})),use_container_width=True)
        t=df.assign(Hour=((df.Time/3600)%24).astype(int),Type=df.Class.map({0:'Genuine',1:'Fraud'})).groupby(['Hour','Type']).size().reset_index(name='Transactions'); st.plotly_chart(theme(px.line(t,x='Hour',y='Transactions',color='Type',markers=True,color_discrete_map={'Genuine':'#3988ff','Fraud':'#ff5575'})),use_container_width=True)
        st.caption('Dataset-derived statistics only; this is not live bank traffic.')
elif page=='Fraud Detection':
    hero('TRANSACTION LAB','Analyze a transaction','Choose a historical row or enter all anonymized model features.')
    if model is None: empty()
    else:
        mode=st.radio('Input method',['Choose sample dataset row','Manual feature entry'],horizontal=True)
        if mode.startswith('Choose'):
            idx=st.number_input('Dataset row index',0,len(df)-1,0); row=df.iloc[int(idx)]; st.write(f"Record #{int(idx)} · Amount ${row.Amount:.2f}")
            if st.button('RUN FRAUD ANALYSIS',use_container_width=True): st.session_state['result']=(predict_one(model,row[FEATURE_COLUMNS].to_dict()),int(row.Class))
        else:
            with st.form('manual'):
                amount=st.number_input('Amount ($)',min_value=0.0,value=100.0); tm=st.number_input('Time (seconds)',min_value=0.0,value=0.0)
                vals={'Amount':amount,'Time':tm}
                with st.expander('Anonymized features V1–V28'):
                    cols=st.columns(4)
                    for i in range(1,29): vals[f'V{i}']=cols[(i-1)%4].number_input(f'V{i}',value=0.0,key=f'manv{i}')
                go=st.form_submit_button('RUN FRAUD ANALYSIS')
            if go: st.session_state['result']=(predict_one(model,vals),None)
        if 'result' in st.session_state:
            res,actual=st.session_state['result']; a,b,c=st.columns(3); a.metric('MODEL DECISION','FRAUD' if res['prediction'] else 'GENUINE'); b.metric('FRAUD PROBABILITY',f"{res['probability']:.1%}"); c.metric('RISK LEVEL',res['risk'])
            (st.error if res['prediction'] else st.success)('Potential fraud flagged — review required.' if res['prediction'] else 'Classified as likely genuine; not a safety guarantee.')
            fig=go.Figure(go.Indicator(mode='gauge+number',value=res['probability']*100,title={'text':'Estimated fraud probability'},number={'suffix':'%'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#36d6e7'},'steps':[{'range':[0,35],'color':'#153b45'},{'range':[35,70],'color':'#55451d'},{'range':[70,100],'color':'#54243a'}]})); st.plotly_chart(theme(fig,270),use_container_width=True)
            if actual is not None: st.caption(f'Dataset ground-truth label: {"Fraud" if actual else "Genuine"}')
elif page=='Batch Analysis':
    hero('BATCH INTELLIGENCE','Screen a CSV batch','Upload a CSV containing Time, V1–V28 and Amount.')
    if model is None: empty()
    else:
        batch=st.file_uploader('Upload transactions CSV',type='csv',key='batch')
        if batch:
            try:
                bd=pd.read_csv(batch); missing=[x for x in FEATURE_COLUMNS if x not in bd]
                if missing: st.error('Missing columns: '+', '.join(missing))
                elif st.button('ANALYZE BATCH'):
                    p=model.predict_proba(bd[FEATURE_COLUMNS])[:,1]; out=bd.copy(); out['Fraud_Probability']=p; out['Prediction']=np.where(p>=.5,'Fraud','Genuine'); st.session_state['batchout']=out
            except Exception as e: st.error(str(e))
        if 'batchout' in st.session_state:
            out=st.session_state['batchout']; n=int((out.Prediction=='Fraud').sum()); a,b,c=st.columns(3); a.metric('ROWS ANALYZED',len(out)); b.metric('FLAGGED',n); c.metric('FLAG RATE',f'{n/max(len(out),1):.1%}')
            st.plotly_chart(theme(px.histogram(out,x='Fraud_Probability',color='Prediction',nbins=30,color_discrete_map={'Genuine':'#3988ff','Fraud':'#ff5575'})),use_container_width=True); st.dataframe(out.head(100),use_container_width=True); st.download_button('Download prediction report',out.to_csv(index=False).encode(), 'fraudshield_report.csv','text/csv')
elif page=='Model Performance':
    hero('MODEL INTELLIGENCE','Measure what matters','Imbalanced fraud datasets require more than accuracy.')
    if metrics is None: empty()
    else:
        a,b,c,d=st.columns(4)
        for col,label,key in [(a,'Accuracy','accuracy'),(b,'Precision','precision'),(c,'Recall','recall'),(d,'F1 Score','f1')]: col.metric(label,f'{metrics[key]:.2%}')
        cm=metrics['cm']; fig=px.imshow(cm,text_auto=True,x=['Predicted Genuine','Predicted Fraud'],y=['Actual Genuine','Actual Fraud'],color_continuous_scale='Blues',title='Confusion Matrix'); st.plotly_chart(theme(fig,380),use_container_width=True)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=metrics['fpr'],y=metrics['tpr'],mode='lines',name=f"ROC AUC {metrics['auc']:.3f}",line={'color':'#36d6e7','width':3})); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',name='Random baseline',line={'dash':'dash'})); fig.update_layout(xaxis_title='False positive rate',yaxis_title='True positive rate'); st.plotly_chart(theme(fig),use_container_width=True)
        st.caption('Held-out stratified test split; threshold 0.50.')
elif page=='Dataset Explorer':
    hero('DATA INTELLIGENCE','Know your data','Explore class balance, distributions, feature relationships and data quality.')
    if df is None: empty()
    else:
        a,b,c=st.columns(3); a.metric('ROWS',f'{len(df):,}'); b.metric('COLUMNS',len(df.columns)); c.metric('MISSING CELLS',int(df.isna().sum().sum()))
        t1,t2,t3=st.tabs(['Distribution','Feature analysis','Data quality'])
        with t1:
            sm=df.sample(min(15000,len(df)),random_state=4).copy(); sm['Type']=sm.Class.map({0:'Genuine',1:'Fraud'}); st.plotly_chart(theme(px.histogram(sm,x='Amount',color='Type',nbins=60,barmode='overlay',opacity=.7)),use_container_width=True); st.dataframe(df.Class.value_counts().rename(index={0:'Genuine',1:'Fraud'}),use_container_width=True)
        with t2:
            feat=st.selectbox('Feature',['Amount','Time']+[f'V{i}' for i in range(1,29)]); sm=df.sample(min(20000,len(df)),random_state=2); st.plotly_chart(theme(px.box(sm,x='Class',y=feat,color='Class',points=False)),use_container_width=True); corr=df[FEATURE_COLUMNS+['Class']].corr(numeric_only=True); st.plotly_chart(theme(px.imshow(corr,color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title='Correlation heatmap'),560),use_container_width=True)
        with t3: st.dataframe(pd.DataFrame({'Type':df.dtypes.astype(str),'Missing':df.isna().sum(),'Unique':df.nunique()}),use_container_width=True); st.dataframe(df.head(20),use_container_width=True)
elif page=='How It Works':
    hero('LEARN / MODEL PIPELINE','From transaction to signal','A transparent walkthrough of the classification workflow.')
    for n,title,desc in [('01','Dataset','Historical transactions labelled genuine (0) or fraud (1).'),('02','Preprocessing','Validate columns, stratified split and scale Amount/Time and features.'),('03','Training','Train a class-weight-balanced Logistic Regression classifier.'),('04','Evaluation','Review precision, recall, F1, ROC-AUC and confusion matrix.'),('05','Prediction','Estimate fraud probability and apply a 0.50 threshold.')]:
        with st.container(border=True): st.markdown(f'### <span style="color:#36d6e7">{n}</span> · {title}',unsafe_allow_html=True); st.write(desc)
    st.info("V1–V28 are anonymized features. This educational prototype is not a production bank fraud-control system; a prediction is not proof of fraud.")
else:
    hero('PROJECT / INFORMATION','About FraudShield AI','A student case study demonstrating supervised machine learning for transaction fraud classification.')
    st.subheader('Project objective'); st.write('Build an understandable end-to-end application to explore historical card transactions, train a baseline classifier and visualize predictions and evaluation results.')
    st.subheader('Technology stack'); st.write('Python · Streamlit · Pandas · NumPy · Scikit-learn · Plotly')
    st.subheader('Dataset'); st.write('Kaggle Credit Card Fraud Detection dataset (European cardholders, September 2013), with anonymized PCA features and binary Class label.'); st.markdown('[Open dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)')
    st.warning('Academic demonstration only. Not intended for real financial decisions or payment authorization.')
