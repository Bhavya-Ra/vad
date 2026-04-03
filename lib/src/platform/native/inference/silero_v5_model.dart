// lib/src/platform/native/inference/silero_v5_model.dart
// ignore_for_file: public_member_api_docs, avoid_print

// Dart imports:
import 'dart:io';
import 'dart:typed_data';

// Flutter imports:
import 'package:flutter/services.dart';

// Package imports:
import 'package:path_provider/path_provider.dart';

// Project imports:
import 'package:vad/src/core/vad_event.dart';
import 'package:vad/src/core/vad_model.dart';
import 'package:vad/src/utils/model_utils.dart';
import 'package:vad/src/platform/native/onnxruntime/ort_session.dart';
import 'package:vad/src/platform/native/onnxruntime/ort_value.dart';
import 'package:vad/src/platform/native/onnxruntime/ort_threading_config.dart';

class SileroV5Model implements VadModel {
  final OrtSession _session;
  final OrtSessionOptions _sessionOptions;
  final int _sampleRate;

  static const int _batch = 1;
  var _state = List.filled(
      2, List.filled(1, Float32List.fromList(List.filled(128, 0.0))));

  SileroV5Model._(
    this._session,
    this._sessionOptions,
    this._sampleRate,
  ) {
    resetState();
  }

  static Future<SileroV5Model> create(
    String modelPath,
    int sampleRate,
    bool isDebug, {
    OrtThreadingConfig? threadingConfig,
  }) async {
    try {
      final config = threadingConfig ?? OrtThreadingConfig.platformOptimal();
      final sessionOptions = OrtSessionOptions()
        ..setInterOpNumThreads(config.interOpNumThreads)
        ..setIntraOpNumThreads(config.intraOpNumThreads)
        ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

      final bytes = await _loadModelBytes(modelPath);
      final session = OrtSession.fromBuffer(bytes, sessionOptions);

      if (isDebug) {
        print('SileroV5Model initialized from $modelPath');
        print('Model input names: ${session.inputNames}');
        print('Model output names: ${session.outputNames}');
        print(
            'Threading config: intraOp=${config.intraOpNumThreads}, interOp=${config.interOpNumThreads}');
      }

      return SileroV5Model._(session, sessionOptions, sampleRate);
    } catch (e) {
      print('Error creating SileroV5Model: $e');
      rethrow;
    }
  }

  @override
  void resetState() {
    _state = List.filled(
        2, List.filled(1, Float32List.fromList(List.filled(128, 0.0))));
  }

  @override
  Future<SpeechProbabilities> process(Float32List frame) async {
    final inputNames = getModelInputNames('v5');
    final outputIndices = getModelOutputIndices('v5');

    final inputOrt =
        OrtValueTensor.createTensorWithDataList(frame, [_batch, frame.length]);
    final srOrt = OrtValueTensor.createTensorWithData(_sampleRate);
    final stateOrt = OrtValueTensor.createTensorWithDataList(_state);
    final runOptions = OrtRunOptions();

    final inputs = {
      inputNames['input']!: inputOrt,
      inputNames['sr']!: srOrt,
      inputNames['state']!: stateOrt,
    };

    final outputs = _session.run(runOptions, inputs);

    // Safe to release inputs now - isolate creates its own copies
    inputOrt.release();
    srOrt.release();
    stateOrt.release();
    runOptions.release();

    final outputIdx = outputIndices['output']!;
    final stateIdx = outputIndices['state']!;

    // Check if we have the expected number of outputs
    if (outputs.length < 2 ||
        outputs[outputIdx] == null ||
        outputs[stateIdx] == null) {
      throw Exception(
          'Invalid model outputs: expected 2 outputs but got ${outputs.length}');
    }

    final speechProb = (outputs[outputIdx]!.value as List<List<double>>)[0][0];

    // Deep copy the state to avoid memory issues across isolates
    final stateValue = outputs[stateIdx]!.value as List<List<List<double>>>;
    _state = List.generate(
      stateValue.length,
      (i) => List.generate(
        stateValue[i].length,
        (j) => Float32List.fromList(List<double>.from(stateValue[i][j])),
      ),
    );

    // Don't release outputs - they're already released in the isolate

    return SpeechProbabilities(
      isSpeech: speechProb,
      notSpeech: 1.0 - speechProb,
    );
  }

  @override
  Future<void> release() async {
    _sessionOptions.release();
    _session.release();
  }

  static Future<Uint8List> _loadModelBytes(String modelPath) async {
    if (modelPath.startsWith('http://') || modelPath.startsWith('https://')) {
      // Check disk cache first — avoids re-downloading on every app session.
      final cacheFile = await _modelCacheFile(modelPath);
      if (await cacheFile.exists()) {
        return await cacheFile.readAsBytes();
      }
      // Download and persist to disk for future sessions.
      final client = HttpClient();
      try {
        final request = await client.getUrl(Uri.parse(modelPath));
        final response = await request.close();
        if (response.statusCode == 200) {
          final builder = BytesBuilder();
          await for (final chunk in response) {
            builder.add(chunk);
          }
          final bytes = builder.toBytes();
          await cacheFile.writeAsBytes(bytes);
          return bytes;
        } else {
          throw Exception(
              'HTTP ${response.statusCode}: Failed to download model from $modelPath');
        }
      } finally {
        client.close();
      }
    } else if (modelPath.startsWith('file://')) {
      // Load from device file system (explicit path override).
      final filePath = modelPath.replaceFirst('file://', '');
      return await File(filePath).readAsBytes();
    } else {
      // Load from Flutter asset bundle.
      final rawAssetFile = await rootBundle.load(modelPath);
      return rawAssetFile.buffer.asUint8List();
    }
  }

  /// Returns a stable cache [File] for a given CDN URL.
  /// Uses the URL's last path segment as the filename (e.g. silero_vad_v5.onnx).
  static Future<File> _modelCacheFile(String url) async {
    final dir = await getApplicationSupportDirectory();
    final fileName = Uri.parse(url).pathSegments.last;
    return File('${dir.path}/$fileName');
  }
}
