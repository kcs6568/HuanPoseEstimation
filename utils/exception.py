class CropSettingError(Exception):
    def __init__(self, message):
        self.message = message
        super(CropSettingError, self).__init__(message)


class ErrorCountingNotMatchError(Exception):
    def __init__(self, message):
        self.message = message
        super(ErrorCountingNotMatchError, self).__init__(message)


class ImageDataNotLoadedError(Exception):
    def __init__(self, message):
        self.message = message
        super(ImageDataNotLoadedError, self).__init__(message)


