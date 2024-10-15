// Copyright (c)  2024  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class InfoScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    const double height = 20;
    return Container(
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text('Everything is open-sourced.'),
            SizedBox(height: height),
            InkWell(
              child: Text('Code: https://github.com/k2-fsa/sherpa-onnx'),
              onTap: () => launch('https://k2-fsa.github.io/sherpa/onnx/'),
            ),
            SizedBox(height: height),
            InkWell(
              child: Text('Doc: https://k2-fsa.github.io/sherpa/onnx/'),
              onTap: () => launch('https://k2-fsa.github.io/sherpa/onnx/'),
            ),
            SizedBox(height: height),
            Text('QQ 群: 744602236'),
            SizedBox(height: height),
            InkWell(
              child: Text(
                  '微信群: https://k2-fsa.github.io/sherpa/social-groups.html'),
              onTap: () =>
                  launch('https://k2-fsa.github.io/sherpa/social-groups.html'),
            ),
          ],
        ),
      ),
    );
  }
}
