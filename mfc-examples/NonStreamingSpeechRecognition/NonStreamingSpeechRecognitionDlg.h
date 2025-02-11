
// NonStreamingSpeechRecognitionDlg.h : header file
//

#pragma once

#include <string>
#include <vector>

#include "portaudio.h"
#include "sherpa-onnx/c-api/c-api.h"

class Microphone {
 public:
  Microphone();
  ~Microphone();
};

// CNonStreamingSpeechRecognitionDlg dialog
class CNonStreamingSpeechRecognitionDlg : public CDialogEx {
  // Construction
 public:
  CNonStreamingSpeechRecognitionDlg(
      CWnd *pParent = nullptr);  // standard constructor
  ~CNonStreamingSpeechRecognitionDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
  enum { IDD = IDD_NONSTREAMINGSPEECHRECOGNITION_DIALOG };
#endif

 protected:
  virtual void DoDataExchange(CDataExchange *pDX);  // DDX/DDV support

  // Implementation
 protected:
  HICON m_hIcon;

  // Generated message map functions
  virtual BOOL OnInitDialog();
  afx_msg void OnPaint();
  afx_msg HCURSOR OnQueryDragIcon();
  DECLARE_MESSAGE_MAP()
 public:
  afx_msg void OnBnClickedOk();
  int RunThread();

 private:
  Microphone mic_;

  const SherpaOnnxOfflineRecognizer *recognizer_ = nullptr;
  SherpaOnnxOfflineRecognizerConfig config_;

  PaStream *pa_stream_ = nullptr;
  CButton my_btn_;
  CEdit my_text_;
  std::vector<std::string> results_;

 public:
  bool started_ = false;
  std::vector<float> samples_;

 private:
  void AppendTextToEditCtrl(const std::string &s);
  void AppendLineToMultilineEditCtrl(const std::string &s);
  void InitMicrophone();

  bool Exists(const std::string &filename);
  void InitRecognizer();

  void InitParaformer();
  void InitWhisper();
  void ShowInitRecognizerHelpMessage();
};
