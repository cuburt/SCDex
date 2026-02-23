from dashboard import init_dashboard


app = init_dashboard(None)


if __name__ == '__main__':
    app.run(debug=True)