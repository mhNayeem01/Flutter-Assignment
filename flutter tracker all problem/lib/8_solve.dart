
import 'package:flutter/material.dart';

void main() {
  runApp(const DrawerNavigationApp());
}

class DrawerNavigationApp extends StatelessWidget {
  const DrawerNavigationApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Navigation Drawer',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 30,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.blue,
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(color: Colors.blue),
              child: Text(
                'Menu',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 25,
                ),
              ),
            ),
            buildDrawerItem(context, Icons.home, 'Home', const HomePage()),
            buildDrawerItem(context, Icons.settings, 'Settings', const SettingsPage()),
            buildDrawerItem(context, Icons.info, 'About', const AboutPage()),
            buildDrawerItem(context, Icons.person, 'Profile', const ProfilePage()),
            buildDrawerItem(context, Icons.contact_mail, 'Contact', const ContactPage()),
            buildDrawerItem(context, Icons.help, 'Help', const HelpPage()),
          ],
        ),
      ),
      body: const Center(
        child: Text(
          'Press the THREE bar to open the drawer',
          style: TextStyle(fontSize: 18),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }

  ListTile buildDrawerItem(
      BuildContext context, IconData icon, String title, Widget page) {
    return ListTile(
      leading: Icon(icon),
      title: Text(title),
      onTap: () {
        Navigator.pop(context);
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => page),
        );
      },
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'Home Page', message: 'Welcome to the Home Page');
  }
}

class SettingsPage extends StatelessWidget {
  const SettingsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'Settings', message: 'This is the Settings Page');
  }
}

class AboutPage extends StatelessWidget {
  const AboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'About', message: 'This is the About Page');
  }
}

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'Profile', message: 'This is the Profile Page');
  }
}

class ContactPage extends StatelessWidget {
  const ContactPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'Contact', message: 'This is the Contact Page');
  }
}

class HelpPage extends StatelessWidget {
  const HelpPage({super.key});

  @override
  Widget build(BuildContext context) {
    return const SimplePage(title: 'Help', message: 'This is the Help Page');
  }
}

class SimplePage extends StatelessWidget {
  final String title;
  final String message;

  const SimplePage({super.key, required this.title, required this.message});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 30,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.blue,
      ),
      body: Center(
        child: Text(
          message,
          style: const TextStyle(fontSize: 20),
        ),
      ),
    );
  }
}
