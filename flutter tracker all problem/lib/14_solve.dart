import 'package:flutter/material.dart';
import 'package:flutter_slidable/flutter_slidable.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Swipeable List App',
      theme: ThemeData(
        primarySwatch: Colors.deepPurple,
      ),
      home: const SwipeableListScreen(),
    );
  }
}

class SwipeableListScreen extends StatefulWidget {
  const SwipeableListScreen({super.key});

  @override
  State<SwipeableListScreen> createState() => _SwipeableListScreenState();
}

class _SwipeableListScreenState extends State<SwipeableListScreen> {
  List<String> items = List.generate(10, (index) => 'Item ${index + 1}');

  void _editItem(int index) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('Edit pressed on ${items[index]}')),
    );
  }

  void _deleteItem(int index) {
    setState(() {
      items.removeAt(index);
    });

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Item deleted')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Swipe Actions',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 28),
        ),
        centerTitle: true,
        backgroundColor: Colors.deepPurple,
      ),
      body: ListView.builder(
        itemCount: items.length,
        padding: const EdgeInsets.all(10),
        itemBuilder: (context, index) {
          return Container(
            margin: const EdgeInsets.symmetric(vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [
                BoxShadow(color: Colors.grey, blurRadius: 4, offset: Offset(2, 4)),
              ],
            ),
            child: Slidable(
              key: ValueKey(items[index]),
              endActionPane: ActionPane(
                motion: const ScrollMotion(),
                children: [
                  SlidableAction(
                    onPressed: (_) => _editItem(index),
                    backgroundColor: Colors.blue,
                    icon: Icons.edit,
                    label: 'Edit',
                  ),
                  SlidableAction(
                    onPressed: (_) => _deleteItem(index),
                    backgroundColor: Colors.red,
                    icon: Icons.delete,
                    label: 'Delete',
                  ),
                ],
              ),
              child: ListTile(
                title: Text(items[index]),
                leading: const Icon(Icons.list),
              ),
            ),
          );
        },
      ),
    );
  }
}
