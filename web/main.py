import streamlit as st
import pandas as pd
import numpy as np
import pickle
import catboost

st.set_page_config(
        page_title="GibData",
)
def upload():
    uploaded_file = st.file_uploader("Выберите файл")
    if uploaded_file is not None:
        format_name = uploaded_file.name.split('.')[1]

        if format_name == "parquet":
            dataframe = pd.read_parquet(uploaded_file)
        elif format_name == "csv":
            dataframe = pd.read_csv(uploaded_file)
        else:
            st.write("Можно загрузить только файлы формата parquet или csv.")
            st.write("Попробуйте снова!")
            return 0
        st.write(dataframe)
        if st.button('Получить результаты'):
            filename = 'model' 
            
            block_col = ["Содержание", "Крайний срок", "Дата обращения", "Дата восстановления",
            "Дата закрытия обращения", "Тип обращения итоговый", "Тип переклассификации", "id"]


            loaded_model = pickle.load(open(filename, 'rb'))
            df = dataframe.drop(block_col, axis = 1)
            df.rename(columns = {'Решение             ':'Решение'}, inplace = True )

            #st.write(df)

            predict = loaded_model.predict(df)
            submission = pd.read_csv("submission.csv")
            submission['Тип переклассификации'] = predict
            submission['Тип обращения на момент подачи'] = df['Тип обращения на момент подачи']
            submission['Тип обращения итоговый'] = np.where(submission['Тип переклассификации'] == 0, submission['Тип обращения на момент подачи'], 
                np.where(submission['Тип переклассификации'] == 1, "Инцидент", "Запрос"))
            submission.drop("Тип обращения на момент подачи", axis = 1, inplace = True)
            st.write(submission)
            def convert_df(df):
                return df.to_csv(index = False)

            csv = convert_df(submission)

            st.download_button(
                label="Скачать данные в формате CSV",
                data=csv,
                file_name='submission.csv',
                mime='text/csv')


if __name__ == "__main__":
       upload()
