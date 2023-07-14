
// NonStreamingSpeechRecognition.h : main header file for the PROJECT_NAME
// application
//

#pragma once

#ifndef __AFXWIN_H__
#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"  // main symbols

// CNonStreamingSpeechRecognitionApp:
// See NonStreamingSpeechRecognition.cpp for the implementation of this class
//

class CNonStreamingSpeechRecognitionApp : public CWinApp {
 public:
  CNonStreamingSpeechRecognitionApp();

  // Overrides
 public:
  virtual BOOL InitInstance();

  // Implementation

  DECLARE_MESSAGE_MAP()
};

extern CNonStreamingSpeechRecognitionApp theApp;
