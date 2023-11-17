import sys


class CustomException(Exception):
    def __init__(self,error_msg, error_details:sys):
        self.erroe_msg= error_msg
        _,_,exc_tb= error_details.exc_info()

        self.line_no= exc_tb.tb_lineno
        self.file_name= exc_tb.tb_frame.f_code.co_filename


    def __str__(self):
        return "Error occured in python Script name [{0}] line number [{1}] error message [{2}]".format(self.file_name, self.line_no, str(self.erroe_msg))
    




