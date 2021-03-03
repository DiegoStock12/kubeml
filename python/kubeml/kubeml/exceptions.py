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


class MergeError(KubeMLException):
    def __init__(self, e: Exception = None):
        super(MergeError, self) \
            .__init__(f"Error merging model: {e}", 500)


class DataError(KubeMLException):
    def __init__(self):
        super(DataError, self) \
            .__init__("Data not present in request", 400)


class InvalidFormatError(KubeMLException):
    def __init__(self):
        super(InvalidFormatError, self) \
            .__init__("The data provided is not in an appropriate format", 400)


class StorageError(KubeMLException):
    def __init__(self, e: Exception):
        super(StorageError, self) \
            .__init__(f"Could not access storage service: {str(e)}", 500)


class DatasetNotFoundError(KubeMLException):
    def __init__(self):
        super(DatasetNotFoundError, self) \
            .__init__("Dataset not found in storage service", 404)


class InvalidArgsError(KubeMLException):
    def __init__(self, e: Exception):
        super(InvalidArgsError, self) \
            .__init__(f"Error parsing function arguments: {str(e)}", 500)
