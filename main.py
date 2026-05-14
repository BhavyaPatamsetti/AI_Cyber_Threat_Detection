import os

def main():
    while True:
        print("\nAI Cyber Threat Detection System")
        print("--------------------------------")
        print("1. Train Binary Classification Models")
        print("2. Evaluate Binary Model")
        print("3. Predict Binary Threat")
        print("4. Train Multi-Class Model")
        print("5. Predict Attack Type")
        print("6. Train Anomaly Detection Model")
        print("7. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            os.system("python src/train_model.py")
        elif choice == "2":
            os.system("python src/evaluate_model.py")
        elif choice == "3":
            os.system("python src/predict.py")
        elif choice == "4":
            os.system("python src/train_multiclass_model.py")
        elif choice == "5":
            os.system("python src/predict_multiclass.py")
        elif choice == "6":
            os.system("python src/train_anomaly_model.py")
        elif choice == "7":
            print("Exiting system.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
