// lib/src/utils/model_cache_utils.dart
// ignore_for_file: avoid_print

// Dart imports:
import 'dart:io';
import 'dart:typed_data';

// Package imports:
import 'package:path_provider/path_provider.dart';

/// Returns the on-disk cache [File] for a given CDN URL.
/// Uses the URL's last path segment as the filename (e.g. silero_vad_v5.onnx).
Future<File> getModelCacheFile(String url) async {
  final dir = await getApplicationSupportDirectory();
  final fileName = Uri.parse(url).pathSegments.last;
  return File('${dir.path}/$fileName');
}

/// Downloads [url] and writes the bytes to [cacheFile].
/// Throws on non-200 response or network error — caller decides how to handle.
Future<void> downloadModelToCache(String url, File cacheFile) async {
  final client = HttpClient();
  try {
    final request = await client.getUrl(Uri.parse(url));
    final response = await request.close();
    if (response.statusCode != 200) {
      throw Exception(
          'HTTP ${response.statusCode}: Failed to download model from $url');
    }
    final builder = BytesBuilder();
    await for (final chunk in response) {
      builder.add(chunk);
    }
    await cacheFile.writeAsBytes(builder.toBytes());
  } finally {
    client.close();
  }
}
