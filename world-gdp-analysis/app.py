import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


@st.cache_data
def load_and_clean(path: str):
    # Load CSV then coerce numeric columns (dataset uses comma as decimal separator)
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True)

    # Strip whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # Clean numeric columns: try converting everything except Country and Region
    for col in df.columns:
        if col.lower() in ("country", "region"):
            continue
        # convert to string, replace commas used as decimal separators and thousands separators
        s = df[col].astype(str).str.strip()
        # Replace commas used as decimal separator with dot, and remove spaces
        s = s.str.replace(" ", "", regex=False)
        s = s.str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(s, errors='coerce')

    # Standardize column names for convenience
    df = df.rename(columns={
        'GDP ($ per capita)': 'GDP_per_capita',
        'Population': 'Population',
        'Area (sq. mi.)': 'Area_sq_mi',
    })

    return df


def show_eda(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")

    st.markdown("**Dataset preview**")
    st.dataframe(df.head(50))

    st.markdown("**Summary statistics**")
    st.write(df.describe(include='all'))

    # Region filter for plots
    regions = ['All'] + sorted(df['Region'].dropna().unique().tolist())
    sel_region = st.selectbox("Filter by region", regions)
    if sel_region != 'All':
        plot_df = df[df['Region'] == sel_region]
    else:
        plot_df = df

    st.subheader("GDP per capita distribution")
    fig = px.histogram(plot_df, x='GDP_per_capita', nbins=50, title='GDP per capita (distribution)')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter: GDP vs Literacy")
    if 'Literacy (%)' in plot_df.columns:
        fig2 = px.scatter(plot_df, x='Literacy (%)', y='GDP_per_capita', color='Region', hover_name='Country')
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation heatmap (numeric columns)")
    num = plot_df.select_dtypes(include=[np.number])
    if not num.empty:
        corr = num.corr()
        fig3, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0)
        st.pyplot(fig3)


def train_and_evaluate(df: pd.DataFrame, features, target, model_type='RandomForest', n_estimators=100, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    X_train_proc = pipe.fit_transform(X_train)
    X_test_proc = pipe.transform(X_test)

    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    else:
        model = LinearRegression()

    model.fit(X_train_proc, y_train)
    preds = model.predict(X_test_proc)

    # Use sqrt of MSE for RMSE to support older scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Feature importances (if tree-based)
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    return {
        'model': model,
        'pipeline': pipe,
        'X_test': X_test,
        'y_test': y_test,
        'preds': preds,
        'rmse': rmse,
        'r2': r2,
        'importances': importances,
    }


def main():
    st.set_page_config(page_title='World GDP Analysis & Prediction', layout='wide')
    st.title('World GDP Analysis & Prediction')

    data_path = 'c:\\Users\\DELL\\OneDrive\\Desktop\\Word_GDP_Analysis_Prediction\\countries of the world.csv'

    st.sidebar.header('Data & Options')
    if st.sidebar.button('Reload data'):
        load_and_clean.clear()

    df = load_and_clean(data_path)

    st.sidebar.markdown('---')
    st.sidebar.write('Rows: ', df.shape[0])

    # Quick data download
    if st.sidebar.checkbox('Show raw data (downloadable)'):
        st.download_button('Download cleaned CSV', df.to_csv(index=False).encode('utf-8'), file_name='countries_cleaned.csv')
        st.dataframe(df)

    # EDA
    show_eda(df)

    st.header('GDP Prediction')
    st.markdown('Train a simple model to predict `GDP_per_capita`. Choose features and model.')

    # Default features
    default_features = ['Literacy (%)', 'Phones (per 1000)', 'Population', 'Area_sq_mi', 'Birthrate', 'Deathrate']
    available_features = [c for c in df.columns if c not in ['Country', 'Region', 'GDP_per_capita']]

    chosen = st.multiselect('Select predictor features', options=available_features, default=[f for f in default_features if f in available_features])
    target = 'GDP_per_capita'

    st.sidebar.subheader('Model settings')
    model_type = st.sidebar.selectbox('Model', ['RandomForest', 'LinearRegression'])
    n_estimators = st.sidebar.slider('RandomForest n_estimators', 10, 500, 100)
    test_size = st.sidebar.slider('Test set fraction', 0.1, 0.5, 0.2)

    if st.button('Train model'):
        if len(chosen) == 0:
            st.error('Please select at least one feature')
        else:
            with st.spinner('Training...'):
                res = train_and_evaluate(df, chosen, target, model_type=model_type, n_estimators=n_estimators, test_size=test_size)

            st.success('Training complete')
            st.write('RMSE:', f"{res['rmse']:.2f}")
            st.write('R2:', f"{res['r2']:.3f}")

            # Scatter actual vs predicted
            fig = px.scatter(x=res['y_test'], y=res['preds'], labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
            fig.add_shape(type='line', x0=res['y_test'].min(), x1=res['y_test'].max(), y0=res['y_test'].min(), y1=res['y_test'].max(), line=dict(color='red', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

            if res['importances'] is not None:
                fi = pd.Series(res['importances'], index=chosen).sort_values(ascending=False)
                st.subheader('Feature importances')
                st.bar_chart(fi)

            # Prediction from user input
            st.subheader('Make a prediction (manual input)')
            input_vals = {}
            cols = st.columns(2)
            for i, f in enumerate(chosen):
                with cols[i % 2]:
                    val = st.number_input(f, value=float(df[f].median()) if f in df else 0.0)
                    input_vals[f] = val

            if st.button('Predict for input'):
                X_new = pd.DataFrame([input_vals])
                X_new_proc = res['pipeline'].transform(X_new)
                pred = res['model'].predict(X_new_proc)[0]
                st.metric('Predicted GDP per capita', f"{pred:.2f}")


if __name__ == '__main__':
    main()
