package com.github.jing332.tts_server_android.compose.widgets

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxColors
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.utils.clickableRipple

@Composable
fun TextCheckBox(
    modifier: Modifier = Modifier,
    text: @Composable RowScope.() -> Unit,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
    colors: CheckboxColors = CheckboxDefaults.colors(),

    horizontalArrangement: Arrangement.Horizontal = Arrangement.Center,
    verticalAlignment: Alignment.Vertical = Alignment.CenterVertically,
) {
    Row(
        modifier
            .height(48.dp)
            .clip(MaterialTheme.shapes.small)
            .clickableRipple(role = Role.Checkbox) { onCheckedChange(!checked) },
        verticalAlignment = verticalAlignment,
        horizontalArrangement = horizontalArrangement,
    ) {
        Row(Modifier.padding(horizontal = 8.dp)) {
            Checkbox(colors = colors, checked = checked, onCheckedChange = null)
            text()
        }
    }
}