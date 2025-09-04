import streamlit as st
 

def calculator(operation, num1, num2):
    if operation == "Add":
        return num1 + num2 
    elif operation == "Subtract":
        return num1 - num2
    elif operation == "Multiply":
        return num1 * num2
    elif operation == "Divide":
        if num2 != 0:
            return num1 / num2
        else: 
            return "error: Divisiion by zero"
        
def create_calculator_app():
    st.title ("Simple Calculator")

 # boxes for user input
    num1 = st.number_input("Enter the first number:", value=0.0)
    num2 = st.number_input("Enter the second number:", value=0.0)

    # A dropdown for selecting operation
    operation = st.selectbox("Select operation:", ["Add", "Subtract", "Multiply", "Divide"])

    # Button that performs the calculation 
    if st.button("Calculate"):
        result = calculator(operation, num1, num2)
        st.success(f"Answer: {result}")
 
create_calculator_app()
