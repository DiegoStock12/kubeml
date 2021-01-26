class KubeMLException(Exception):

    def __init__(self, message, status_code):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code

    def to_dict(self):
        return {
            "error": self.message,
            "code": self.status_code
        }


class InvalidDataFormatException(KubeMLException):
    def __init__(self):
        super(InvalidDataFormatException, self) \
            .__init__("The data provided is not in an appropriate format", 400)


class StorageUnreachableException(KubeMLException):
    def __init__(self, e: Exception):
        super(StorageUnreachableException, self) \
            .__init__(f"Could not access storage service: {str(e)}", 500)


class DatasetNotFoundException(KubeMLException):
    def __init__(self):
        super(DatasetNotFoundException, self) \
            .__init__("Dataset not found in storage service", 404)


class InvalidArgsException(KubeMLException):
    def __init__(self, e: Exception):
        super(InvalidArgsException, self) \
            .__init__(f"Error parsing function arguments: {str(e)}", 500)
