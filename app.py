# import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import pickle
import os

# Load the saved model
try:
    # Assuming the saved model is named 'svm_model.pkl'
    # Adjust the filename if your best model is different (e.g., random_forest_model.pkl)
    model_filename = 'svm_model.pkl'
    if not os.path.exists(model_filename):
        st.error(f"Model file not found: {model_filename}")
        st.stop() # Stop the app if the model file is not found

    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    st.success(f"Model '{model_filename}' loaded successfully.")

except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop() # Stop the app if there's an error loading the model

st.title("작업자 행동 분류 웹 애플리케이션")

st.write("CSV 파일을 업로드하여 작업자의 행동(Stop, Walk, Run)을 분류합니다.")

# File uploader for the CSV data
uploaded_file = st.file_uploader("작업자 행동 데이터 CSV 파일을 선택하세요", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df_predict = pd.read_csv(uploaded_file)

        st.subheader("업로드된 데이터 미리보기")
        st.write(df_predict.head())

        # Preprocessing the uploaded data: Similar to the training data preprocessing
        # Ensure 'Absolute' column is handled as a categorical variable
        if 'Absolute' in df_predict.columns:
            df_predict = pd.get_dummies(df_predict, columns=['Absolute'], drop_first=True)
        else:
            st.warning("Uploaded data does not contain an 'Absolute' column. Proceeding without it.")

        # Ensure all columns that the model was trained on are present in the prediction data
        # This is crucial for consistent predictions. We need the list of columns from the training data X.
        # Since we don't have X available directly here, we'll need a way to get the training columns.
        # A robust way is to save the list of columns along with the model.
        # For now, let's assume the columns are consistent or handle missing columns by adding them with 0s.

        # --- IMPORTANT: Need to get the training column names ---
        # The model expects the same columns with the same order as it was trained on.
        # A better approach would be to save the list of training columns with the model.
        # Since that wasn't done in the preceding code, we'll make an assumption or
        # require the user to upload data with the same column structure.
        # A quick fix is to identify columns present in the loaded model's training data
        # This is hacky and relies on internal model structure, not recommended for production.
        # A proper solution involves saving column names during training.

        # Let's try to get feature names from the trained model if available
        try:
            if hasattr(model, 'feature_names_in_'):
                training_columns = model.feature_names_in_
                # Add missing columns with 0 and reorder columns
                for col in training_columns:
                    if col not in df_predict.columns:
                        df_predict[col] = 0
                df_predict = df_predict[training_columns]
            else:
                st.warning("Could not retrieve training column names from the model. Prediction might fail if columns are inconsistent.")
                # Proceeding with available columns, might cause errors if columns don't match
                pass # Allow to proceed, but with a warning

        except Exception as col_error:
            st.error(f"Error aligning columns: {col_error}")
            st.stop()


        # Make predictions
        predictions = model.predict(df_predict)

        st.subheader("예측 결과")

        # Add predictions as a new column to the dataframe for display
        df_predict['예측된_행동'] = predictions

        st.write(df_predict)

        # Display summary of predictions
        st.subheader("예측 결과 요약")
        prediction_summary = df_predict['예측된_행동'].value_counts()
        st.write(prediction_summary)

        # Optional: Visualize the prediction summary
        st.subheader("예측된 행동 분포")
        fig, ax = plt.subplots()
        prediction_summary.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax.set_title('예측된 행동 분포')
        ax.set_xlabel('행동 유형')
        ax.set_ylabel('빈도')
        st.pyplot(fig)


    except pd.errors.EmptyDataError:
        st.error("업로드된 파일이 비어 있습니다.")
    except pd.errors.ParserError:
        st.error("업로드된 파일을 파싱할 수 없습니다. CSV 형식을 확인해주세요.")
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {e}")

