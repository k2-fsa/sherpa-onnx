package com.k2fsa.sherpa.onnx.fa

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.k2fsa.sherpa.onnx.fa.screens.AlignScreen
import com.k2fsa.sherpa.onnx.fa.screens.InfoScreen
import com.k2fsa.sherpa.onnx.fa.ui.theme.SherpaOnnxForcedAlignmentTheme

const val TAG = "sherpa-onnx-fa"

class MainActivity : ComponentActivity() {
    // The manager owns the native aligner; it is created once at app start
    // and kept across tab switches so the model stays loaded.
    private var manager: ForcedAlignmentManager? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val m = ForcedAlignmentManager(applicationContext)
        manager = m
        m.warmUp() // load the model now so the first run is fast
        enableEdgeToEdge()
        setContent {
            SherpaOnnxForcedAlignmentTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    MainScreen(m)
                }
            }
        }
    }

    override fun onDestroy() {
        manager?.dispose()
        manager = null
        super.onDestroy()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(manager: ForcedAlignmentManager, modifier: Modifier = Modifier) {
    val navController = rememberNavController()
    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary,
                ),
                title = {
                    Text(
                        "Next-gen Kaldi: Forced Alignment",
                        fontWeight = FontWeight.Bold,
                    )
                },
            )
        },
        content = { padding ->
            Column(Modifier.padding(padding)) {
                NavigationHost(navController = navController, manager = manager)
            }
        },
        bottomBar = {
            BottomNavigationBar(navController = navController)
        },
    )
}

@Composable
fun NavigationHost(navController: NavHostController, manager: ForcedAlignmentManager) {
    NavHost(navController = navController, startDestination = NavRoutes.Align.route) {
        composable(NavRoutes.Align.route) {
            AlignScreen(manager)
        }
        composable(NavRoutes.Info.route) {
            InfoScreen()
        }
    }
}

@Composable
fun BottomNavigationBar(navController: NavHostController) {
    NavigationBar {
        val backStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = backStackEntry?.destination?.route

        NavBarItems.BarItems.forEach { navItem ->
            NavigationBarItem(
                selected = currentRoute == navItem.route,
                onClick = {
                    navController.navigate(navItem.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                icon = {
                    Icon(imageVector = navItem.image, contentDescription = navItem.title)
                },
                label = {
                    Text(text = navItem.title)
                },
            )
        }
    }
}
