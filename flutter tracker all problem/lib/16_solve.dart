import 'package:flutter/material.dart';
import 'dart:math';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Animated Container Demo',
      theme: ThemeData(primarySwatch: Colors.teal),
      home: const AnimatedContainerScreen(),
    );
  }
}

class AnimatedContainerScreen extends StatefulWidget {
  const AnimatedContainerScreen({super.key});

  @override
  State<AnimatedContainerScreen> createState() => _AnimatedContainerScreenState();
}

class _AnimatedContainerScreenState extends State<AnimatedContainerScreen> {
  double _width = 150;
  double _height = 150;
  Color _color = Colors.teal;
  BorderRadiusGeometry _borderRadius = BorderRadius.circular(20);

  void _changeProperties() {
    setState(() {
      final random = Random();
      _width = 100 + random.nextInt(150).toDouble();
      _height = 100 + random.nextInt(150).toDouble();
      _color = Color.fromRGBO(
        random.nextInt(256),
        random.nextInt(256),
        random.nextInt(256),
        1,
      );
      _borderRadius = BorderRadius.circular(random.nextInt(100).toDouble());
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Animated Box'),
        centerTitle: true,
      ),
      body: Center(
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 600),
          curve: Curves.easeInOut,
          width: _width,
          height: _height,
          decoration: BoxDecoration(
            color: _color,
            borderRadius: _borderRadius,
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _changeProperties,
        child: const Icon(Icons.play_arrow),
      ),
    );
  }
}
