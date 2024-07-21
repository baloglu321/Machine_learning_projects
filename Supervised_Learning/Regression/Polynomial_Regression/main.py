import streamlit as st
from utils import  update_model
import matplotlib.pyplot as plt

def update():
    model,model_status,plot_data = update_model()
    return model,model_status,plot_data


def start_streamlit():
    model,model_status,plot_data = update()
    y_test,y_test_pred,y_train,y_train_pred=plot_data
    train_mse,test_mse,train_r2,test_r2=model_status
    st.header("Elestic net Regression", divider="blue")
    st.write(
        "This page contains an algorithm that predicts the next day's close of nvidia stock based on the previous day's movements using a Elestic net Regression model"
    )
    st.divider()
    st.header(":blue[Model Status]", divider="blue")
    st.caption(f":blue[MSE:{round(test_mse,2)}] and :blue[R2:{round(test_r2,5)}]")
    st.divider()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))


    ax[0].scatter(y_train, y_train_pred, color='blue', edgecolor='w', alpha=0.7)
    ax[0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax[0].set_xlabel('Real Values')
    ax[0].set_ylabel('Prediction Values')
    ax[0].set_title('Training Set: Real vs Prediction')


    ax[1].scatter(y_test, y_test_pred, color='red', edgecolor='w', alpha=0.7)
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax[1].set_xlabel('Real Values')
    ax[1].set_ylabel('Prediction Values')
    ax[1].set_title('Test Set: Real vs Prediction')

    plt.tight_layout()
    st.pyplot(fig)

    st.caption(
        "INFO:l Elastic Net Regression is a combination of Ridge and Lasso Regression. It includes both L1 (Lasso) and L2 (Ridge) regularisation terms. Elastic Net combines the two regularisation terms, allowing both feature selection and minimisation of the coefficients. It is an effective method especially in high-dimensional data sets and in cases of multicollinearity."
    )
    st.divider()
    st.caption(":blue[Improving by MEB]")
    st.divider()


start_streamlit()
