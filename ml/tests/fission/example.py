from flask import current_app


def main():
    current_app.logger.info("Responding to the user")
    return "Hello there", 200
