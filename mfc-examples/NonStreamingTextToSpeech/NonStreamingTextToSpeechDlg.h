
// NonStreamingTextToSpeechDlg.h : header file
//

#pragma once

#include "sherpa-onnx/c-api/c-api.h"

#include <memory>
#include <thread>

#include "portaudio.h"

class Microphone {
 public:
  Microphone();
  ~Microphone();
};

// CNonStreamingTextToSpeechDlg dialog
class CNonStreamingTextToSpeechDlg : public CDialogEx
{
// Construction
public:
	CNonStreamingTextToSpeechDlg(CWnd* pParent = nullptr);	// standard constructor
 ~CNonStreamingTextToSpeechDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_NONSTREAMINGTEXTTOSPEECH_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CEdit my_hint_;
	CEdit speaker_id_;
	CEdit speed_;
	void Init();
	void InitHint();
	CButton generate_btn_;
	afx_msg void OnBnClickedOk();

	SherpaOnnxOfflineTts *tts_ = nullptr;
	CEdit my_text_;
	CEdit output_filename_;

private:
    Microphone mic_;
	std::unique_ptr<std::thread> play_thread_;

};
