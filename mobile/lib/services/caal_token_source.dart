import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:livekit_client/livekit_client.dart' as sdk;

/// Token source that fetches connection details from CAAL's API.
///
/// This replaces LiveKit's SandboxTokenSource with a custom implementation
/// that calls the CAAL frontend's /api/connection-details endpoint.
class CaalTokenSource extends sdk.TokenSource {
  final String baseUrl;

  CaalTokenSource({required this.baseUrl});

  @override
  Future<sdk.TokenSourceResult> fetchToken() async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/connection-details'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'room_config': {
          'agents': [
            {'agent_name': 'CAAL'}
          ]
        }
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to fetch token: ${response.statusCode} ${response.body}');
    }

    final data = jsonDecode(response.body) as Map<String, dynamic>;

    return sdk.TokenSourceResult(
      serverUrl: data['serverUrl'] as String,
      participantToken: data['participantToken'] as String,
    );
  }

  @override
  Future<sdk.TokenSourceResult?> refreshToken(
    String previousToken,
    String serverUrl,
  ) async {
    // CAAL tokens have 15 minute TTL, just fetch a new one
    return fetchToken();
  }

  @override
  Future<sdk.RoomConfiguration?> fetchRoomConfiguration(
    String serverUrl,
    String participantToken,
  ) async {
    // Room configuration is already embedded in the token
    return null;
  }
}
