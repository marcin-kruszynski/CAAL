import 'package:flutter/gestures.dart' show TapGestureRecognizer;
import 'package:flutter/material.dart';
import 'package:livekit_client/livekit_client.dart' as sdk;
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart' show launchUrl;
import '../controllers/app_ctrl.dart' as ctrl;
import '../widgets/button.dart' as buttons;

class WelcomeScreen extends StatelessWidget {
  const WelcomeScreen({super.key});

  @override
  Widget build(BuildContext ctx) => Material(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 36),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              spacing: 30,
              children: [
                // CAAL Logo - using mic icon as placeholder
                Icon(
                  Icons.record_voice_over,
                  size: 80,
                  color: Theme.brightnessOf(ctx) == Brightness.light ? Colors.black : Colors.white,
                ),
                Text(
                  'CAAL',
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Theme.brightnessOf(ctx) == Brightness.light ? Colors.black : Colors.white,
                  ),
                ),
                Text.rich(
                  textAlign: TextAlign.center,
                  TextSpan(
                    children: [
                      const TextSpan(
                        text: 'Start a call to chat with CAAL. Learn more at ',
                      ),
                      TextSpan(
                        text: 'GitHub',
                        style: const TextStyle(
                          color: Colors.blue,
                          decoration: TextDecoration.underline,
                          decorationColor: Colors.blue,
                          decorationThickness: 1,
                        ),
                        recognizer: TapGestureRecognizer()
                          ..onTap = () async {
                            await launchUrl(Uri.parse('https://github.com/CoreWorxLab/CAAL'));
                          },
                      ),
                      const TextSpan(
                        text: '.',
                      ),
                    ],
                  ),
                ),
                // Agent listening indicator
                Consumer<sdk.Session>(
                  builder: (ctx, session, child) => AnimatedOpacity(
                    opacity: session.agent.canListen ? 1.0 : 0.0,
                    duration: const Duration(milliseconds: 300),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.mic,
                            color: Colors.green,
                            size: 18,
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            'CAAL is listening',
                            style: TextStyle(
                              color: Colors.green,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                Consumer2<ctrl.AppCtrl, sdk.Session>(
                  builder: (ctx, appCtrl, session, child) {
                    final isProgressing =
                        appCtrl.isSessionStarting || session.connectionState != sdk.ConnectionState.disconnected;
                    return buttons.Button(
                      text: isProgressing ? 'Connecting' : 'Start call',
                      isProgressing: isProgressing,
                      onPressed: () => appCtrl.connect(),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      );
}
