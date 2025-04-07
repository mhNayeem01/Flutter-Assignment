import 'package:flutter/material.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: CustomDrawerPage(),
    );
  }
}

class CustomDrawerPage extends StatefulWidget {
  const CustomDrawerPage({super.key});

  @override
  State<CustomDrawerPage> createState() => _CustomDrawerPageState();
}

class _CustomDrawerPageState extends State<CustomDrawerPage>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<Offset> _slideAnimation;
  bool _isDrawerOpen = false;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );

    _slideAnimation = Tween<Offset>(
      begin: const Offset(-1.0, 0.0), // hidden left
      end: Offset.zero,               // fully visible
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));
  }

  void _toggleDrawer() {
    setState(() {
      _isDrawerOpen = !_isDrawerOpen;
      _isDrawerOpen ? _controller.forward() : _controller.reverse();
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Widget _buildDrawer() {
    return SlideTransition(
      position: _slideAnimation,
      child: SafeArea(
        child: Container(
          width: 250,
          color: Colors.teal.shade700,
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: const [
              Text('Menu', style: TextStyle(color: Colors.white, fontSize: 24)),
              Divider(color: Colors.white70),
              ListTile(
                leading: Icon(Icons.home, color: Colors.white),
                title: Text('Home', style: TextStyle(color: Colors.white)),
              ),
              ListTile(
                leading: Icon(Icons.settings, color: Colors.white),
                title: Text('Settings', style: TextStyle(color: Colors.white)),
              ),
              ListTile(
                leading: Icon(Icons.logout, color: Colors.white),
                title: Text('Logout', style: TextStyle(color: Colors.white)),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Scaffold(
          appBar: AppBar(
            title: const Text('Custom Drawer'),
            leading: IconButton(
              icon: const Icon(Icons.menu),
              onPressed: _toggleDrawer,
            ),
            backgroundColor: Colors.teal,
          ),
          body: const Center(child: Text('Main Content')),
        ),
        if (_isDrawerOpen) GestureDetector(
          onTap: _toggleDrawer,
          child: Container(color: Colors.black54), // semi-transparent backdrop
        ),
        _buildDrawer(),
      ],
    );
  }
}
