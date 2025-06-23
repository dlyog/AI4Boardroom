# Bookkeeping Flask App

A simple Python Flask application that allows users to upload photos of expense documents, performs OCR to extract text, categorizes the expense using OpenAI's API, and saves the data locally in a JSON file.

## Features

- **Upload Expenses**: Capture and upload expense documents via a mobile-friendly web interface.
- **OCR Processing**: Extract text from images using Tesseract OCR.
- **Expense Categorization**: Automatically categorize expenses for tax purposes using OpenAI Chat Completion API.
- **Data Storage**: Save extracted and categorized data in a JSON file on the local file system.
- **Responsive UI**: Mobile-friendly and responsive user interface.

## Requirements

- **Python 3.x**
- **Packages**:
  - Flask
  - pytesseract
  - openai
  - python-dotenv
  - Pillow
- **Tesseract OCR Engine**: Must be installed on your system.
- **OpenAI API Key**: Stored in a `.env` file.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/bookkeeping.git
   cd bookkeeping

## Test Data

https://expensesreceipt.com/taxi.html