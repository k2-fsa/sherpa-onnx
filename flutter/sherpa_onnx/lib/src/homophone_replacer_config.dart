// Copyright (c)  2025  Xiaomi Corporation

class HomophoneReplacerConfig {
  const HomophoneReplacerConfig(
      {this.dictDir = '', this.lexicon = '', this.ruleFsts = ''});

  factory HomophoneReplacerConfig.fromJson(Map<String, dynamic> json) {
    return HomophoneReplacerConfig(
      dictDir: json['dictDir'] as String? ?? '',
      lexicon: json['lexicon'] as String? ?? '',
      ruleFsts: json['ruleFsts'] as String? ?? '',
    );
  }

  @override
  String toString() {
    return 'HomophoneReplacerConfig(dictDir: $dictDir, lexicon: $lexicon, ruleFsts: $ruleFsts)';
  }

  Map<String, dynamic> toJson() => {
        'dictDir': dictDir,
        'lexicon': lexicon,
        'ruleFsts': ruleFsts,
      };

  final String dictDir;
  final String lexicon;
  final String ruleFsts;
}
