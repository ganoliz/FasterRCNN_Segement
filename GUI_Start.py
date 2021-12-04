# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 14:49:25 2021

@author: Nick
"""

from PyQt5 import QtWidgets

from GUI_Control import GUI_Controller

if __name__=='__main__':
    import sys
    app=QtWidgets.QApplication(sys.argv)
    window=GUI_Controller()
    window.show()
    sys.exit(app.exec_())
